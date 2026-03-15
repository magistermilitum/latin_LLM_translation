#!/usr/bin/env python3
"""CLI for RAG-based Latin -> English translation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

import evaluate
import nltk
import numpy as np
import regex as re
import torch
from bert_score import score as bert_score_fn
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

LAT_STOP = {
    "et", "in", "de", "ad", "per", "cum", "ut", "non", "ne", "autem", "sed",
    "qui", "quae", "quod", "quibus", "quos", "quas", "quoniam", "si",
    "a", "ab", "ex", "pro", "sub", "super", "contra", "inter", "dum", "enim",
    "quia", "quidem", "atque", "nec", "vel", "vero", "iam", "tamen",
}


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embeddings(model_name: str, device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_chroma(persist_dir: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"Chroma persist directory not found: {persist_dir}")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def _nfkd(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def normalize_latin(value: str) -> str:
    value = _nfkd(value)
    value = value.replace("J", "I").replace("j", "i").replace("V", "U").replace("v", "u")
    value = re.sub(r"[^A-Za-z\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.lower()


def toks(value: str) -> List[str]:
    return re.findall(r"[a-z]+", value)


def cheap_stem(word: str) -> str:
    for suffix in (
        "ibus", "orum", "arum", "ium", "ius", "ae", "is", "os", "as", "am",
        "em", "um", "us", "es", "e", "i", "o", "u", "a",
    ):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def content_set(value: str) -> set[str]:
    words = [cheap_stem(w) for w in toks(normalize_latin(value))]
    return {w for w in words if w not in LAT_STOP and len(w) >= 3}


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def retrieve_with_overlap_scored(
    retriever,
    query_lat: str,
    top_k: int,
    max_words: int,
    oversample: int = 35,
    alpha: float = 0.5,
    beta: float = 0.5,
    min_inter: int = 3,
) -> List[str]:
    if top_k == 0:
        return []

    docs = retriever.vectorstore.similarity_search_with_score(query_lat, k=max(top_k * oversample, top_k))
    query_terms = content_set(query_lat)

    scored: List[Tuple[str, float]] = []
    for doc, sim in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        if max_words and max_words > 0:
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words])

        cand_terms = content_set(text)
        inter = len(query_terms & cand_terms)
        if inter < min_inter:
            continue

        jac = inter / max(1, len(query_terms | cand_terms))
        sim_norm = float(sim)
        if sim_norm < -1 or sim_norm > 1:
            sim_norm = 1.0 / (1.0 + sim_norm)
        sim_norm = max(0.0, min(1.0, sim_norm))
        score = alpha * sim_norm + beta * jac
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    seen = set()
    for text, _ in scored:
        if text in seen:
            continue
        out.append(text)
        seen.add(text)
        if len(out) >= top_k:
            break
    return out


def load_nllb(model_name: str, src_lang: str, tgt_lang: str, dtype: str) -> Tuple:
    print(f"Loading NLLB model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    selected_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map, torch_dtype=selected_dtype)
    tokenizer.src_lang = src_lang

    try:
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    except AttributeError:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    if forced_bos_token_id is None or forced_bos_token_id < 0:
        raise ValueError(f"Target language '{tgt_lang}' not recognized by tokenizer.")

    return model, tokenizer, forced_bos_token_id


def nllb_translate_batch(
    model,
    tokenizer,
    texts: List[str],
    forced_bos_id: Optional[int] = None,
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> List[str]:
    if not texts:
        return []
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": True,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }
    if forced_bos_id is not None:
        kwargs["forced_bos_token_id"] = forced_bos_id
    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def load_qwen_chat(model_name: str, load_4bit: bool, lora_path: Optional[str]):
    print(f"Loading Qwen model: {model_name}")
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    if lora_path:
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"LoRA directory not found: {lora_path}")
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def build_refine_messages(
    latin: str,
    nllb_draft: str,
    neighbor_pairs: List[Tuple[str, str]],
    max_example_chars: int,
) -> List[Dict[str, str]]:
    def cut(value: str, n: int) -> str:
        return value if len(value) <= n else value[:n] + "…"

    system_prompt = (
        "You are an expert classicist translator.\n"
        "Task: produce one faithful, polished English translation.\n"
        "Constraints:\n"
        "- Preserve case roles and polarity.\n"
        "- Do not add content not present in the Latin source.\n"
        "Output only the final translation line."
    )

    if neighbor_pairs:
        examples = []
        for i, (lat_nei, en_nei) in enumerate(neighbor_pairs, start=1):
            examples.append(
                f"[EX{i}] LATIN: {cut(lat_nei, max_example_chars)}\n"
                f"[EX{i}] DRAFT: {cut(en_nei, max_example_chars)}"
            )
        examples_text = "\n\n".join(examples)
        user_prompt = (
            "## INSTRUCTION:\n"
            "Revise the draft translation for accuracy and fluency using analogous examples.\n\n"
            f"Latin text:\n{latin}\n\n"
            f"NMT draft (NLLB):\n{nllb_draft}\n\n"
            f"Analogous examples (Latin + NMT draft):\n{examples_text}\n\n"
            "Final translation:"
        )
    else:
        user_prompt = (
            "## INSTRUCTION:\n"
            "Revise the draft translation for accuracy and fluency.\n\n"
            f"Latin text:\n{latin}\n\n"
            f"NMT draft (NLLB):\n{nllb_draft}\n\n"
            "Final translation:"
        )

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


@torch.inference_mode()
def refine_with_qwen(
    qwen_model,
    qwen_tok,
    latin: str,
    nllb_draft: str,
    neighbor_pairs: List[Tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_example_chars: int,
) -> str:
    messages = build_refine_messages(latin, nllb_draft, neighbor_pairs, max_example_chars)
    prompt = qwen_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = qwen_tok([prompt], return_tensors="pt").to(qwen_model.device)

    out = qwen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=qwen_tok.eos_token_id,
    )
    text = qwen_tok.batch_decode(out, skip_special_tokens=True)[0]
    if text.startswith(prompt):
        text = text[len(prompt):]
    if "Final translation:" in text:
        text = text.split("Final translation:")[-1]
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if "assistant" in text:
        text = text.split("assistant")[-1]
    return text.strip().strip('"')


def run_free_mode(args):
    if not args.text and not args.input_file:
        print("In free mode you must pass --text or --input_file.")
        sys.exit(1)

    embeddings = load_embeddings(args.embedding_model, args.embed_device)
    vectordb = load_chroma(args.persist_dir, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": args.top_k})

    nllb_model, nllb_tok, forced_bos = load_nllb(args.nllb_model, args.src_lang, args.tgt_lang, args.nllb_dtype)

    nllb_cache: Dict[str, str] = {}

    def nllb_cached(value: str) -> str:
        if value in nllb_cache:
            return nllb_cache[value]
        nllb_cache[value] = nllb_translate_batch(
            nllb_model,
            nllb_tok,
            [value],
            forced_bos_id=forced_bos,
            max_new_tokens=args.nllb_max_new_tokens,
            num_beams=args.nllb_beams,
        )[0]
        return nllb_cache[value]

    def prepare_item(latin: str) -> Dict:
        neighbors = retrieve_with_overlap_scored(
            retriever,
            latin,
            top_k=args.top_k,
            max_words=args.neighbor_max_words,
        )
        draft = nllb_cached(latin)
        neighbor_pairs = [(text, nllb_cached(text)) for text in neighbors]
        return {"latin": latin, "draft": draft, "neighbors": neighbor_pairs}

    items = []
    if args.text:
        items = [prepare_item(args.text.strip())]
    else:
        with open(args.input_file, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                latin = line.strip()
                if latin:
                    items.append(prepare_item(latin))

    del nllb_model, nllb_tok, embeddings, vectordb, retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qwen_model, qwen_tok = load_qwen_chat(args.qwen_model, not args.no_4bit, args.qwen_lora_path)
    qwen_model.eval()

    outputs = []
    for item in tqdm(items, desc="Qwen refinement"):
        final = refine_with_qwen(
            qwen_model,
            qwen_tok,
            item["latin"],
            item["draft"],
            item["neighbors"],
            args.qwen_max_new_tokens,
            args.qwen_temperature,
            args.qwen_top_p,
            args.max_example_chars,
        )
        record = {
            "latin_text": item["latin"],
            "nllb_draft": item["draft"],
            "neighbors": [{"latin": lt, "nllb_draft": en} for lt, en in item["neighbors"]],
            "final_translation": final,
        }
        outputs.append(record)
        print(f"\nLATIN: {item['latin']}\nNLLB draft: {item['draft']}\nFINAL: {final}\n")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as out_file:
            for row in outputs:
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval_mode(args):
    print("Loading evaluation dataset...")
    ds = load_dataset(args.eval_dataset, split=args.eval_split)
    ds = ds.shuffle(seed=args.seed).select(range(min(args.max_examples, len(ds))))

    latin_list = ds[args.eval_src_col]
    refs_list = ds[args.eval_ref_col]

    embeddings = load_embeddings(args.embedding_model, args.embed_device)
    vectordb = load_chroma(args.persist_dir, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": args.top_k})

    all_neighbor_texts = [
        [
            text for text in unique_keep_order(
                retrieve_with_overlap_scored(
                    retriever,
                    latin,
                    top_k=args.top_k,
                    max_words=args.neighbor_max_words,
                    oversample=args.oversample,
                    alpha=args.alpha,
                    beta=args.beta,
                    min_inter=args.min_inter,
                )
            )
            if text != latin
        ]
        for latin in tqdm(latin_list, desc="Retrieving neighbors")
    ]

    lengths = [len(n) for n in all_neighbor_texts]
    print(
        f"[RAG] neighbors per sample min/median/max = {min(lengths)}/{int(np.median(lengths))}/{max(lengths)}"
    )

    del embeddings, vectordb, retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nllb_model, nllb_tok, forced_bos = load_nllb(args.nllb_model, args.src_lang, args.tgt_lang, args.nllb_dtype)

    unique_texts = set(latin_list)
    for neighbors in all_neighbor_texts:
        unique_texts.update(neighbors)

    nllb_cache: Dict[str, str] = {}
    unique_list = list(unique_texts)
    for i in tqdm(range(0, len(unique_list), args.nllb_batch_size), desc="NLLB Drafting"):
        batch = unique_list[i : i + args.nllb_batch_size]
        translations = nllb_translate_batch(
            nllb_model,
            nllb_tok,
            batch,
            forced_bos_id=forced_bos,
            max_new_tokens=args.nllb_max_new_tokens,
            num_beams=args.nllb_beams,
        )
        for text, translation in zip(batch, translations):
            nllb_cache[text] = translation

    prepared_data = []
    for i, latin in enumerate(latin_list):
        prepared_data.append(
            {
                "latin": latin,
                "draft": nllb_cache[latin],
                "neighbors": [(n, nllb_cache[n]) for n in all_neighbor_texts[i]],
            }
        )

    del nllb_model, nllb_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qwen_model, qwen_tok = load_qwen_chat(args.qwen_model, not args.no_4bit, args.qwen_lora_path)
    qwen_model.eval()

    preds = [
        refine_with_qwen(
            qwen_model,
            qwen_tok,
            item["latin"],
            item["draft"],
            item["neighbors"],
            args.qwen_max_new_tokens,
            args.qwen_temperature,
            args.qwen_top_p,
            args.max_example_chars,
        )
        for item in tqdm(prepared_data, desc="Qwen refinement")
    ]

    del qwen_model, qwen_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    refs_wrapped = [[r] for r in refs_list]
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    meteor = evaluate.load("meteor")

    bleu_res = sacrebleu.compute(predictions=preds, references=refs_wrapped)
    chrf_res = chrf.compute(predictions=preds, references=refs_wrapped, word_order=2)
    meteor_res = meteor.compute(predictions=preds, references=refs_list)
    bert_p, bert_r, bert_f1 = bert_score_fn(
        preds,
        refs_list,
        lang="en",
        verbose=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    print(f"SacreBLEU: {bleu_res['score']:.2f}")
    print(f"chrF++: {chrf_res['score']:.2f}")
    print(f"METEOR: {meteor_res['meteor']:.4f}")
    print(f"BERTScore F1: {bert_f1.mean().item():.4f}")

    if args.compute_comet:
        print("Loading COMET model (Unbabel/wmt22-comet-da)...")
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_model_path)
        comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(latin_list, preds, refs_list)]
        comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        print(f"COMET: {comet_output.system_score:.4f}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as out_file:
            for i, pred in enumerate(preds):
                out_file.write(
                    json.dumps(
                        {
                            "latin_text": latin_list[i],
                            "reference": refs_list[i],
                            "nllb_draft": prepared_data[i]["draft"],
                            "neighbors": [
                                {"latin": lt, "nllb_draft": en} for lt, en in prepared_data[i]["neighbors"]
                            ],
                            "final_translation": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Latin->English translation with Chroma retrieval + NLLB draft + Qwen refinement"
    )
    parser.add_argument("--persist_dir", required=True, help="Path to Chroma persist directory")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--embed_device", default=_default_device())
    parser.add_argument("-k", "--top_k", type=int, default=5)
    parser.add_argument("--neighbor_max_words", type=int, default=100)
    parser.add_argument("--max_example_chars", type=int, default=2000)

    parser.add_argument("--nllb_model", required=True, help="HF model ID or local path for NLLB checkpoint")
    parser.add_argument("--src_lang", default="lat_Latn")
    parser.add_argument("--tgt_lang", default="eng_Latn")
    parser.add_argument("--nllb_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--nllb_max_new_tokens", type=int, default=256)
    parser.add_argument("--nllb_beams", type=int, default=4)
    parser.add_argument("--nllb_batch_size", type=int, default=16)

    parser.add_argument("--qwen_model", required=True, help="HF model ID or local path for Qwen model")
    parser.add_argument("--qwen_lora_path", default=None, help="Optional path to LoRA adapter")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=256)
    parser.add_argument("--qwen_temperature", type=float, default=0.1)
    parser.add_argument("--qwen_top_p", type=float, default=1.0)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading for Qwen")

    parser.add_argument("--mode", choices=["free", "eval"], default="free")
    parser.add_argument("--text", default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)

    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_dataset", default="magistermilitum/latin-french_chartes")
    parser.add_argument("--eval_split", default="train")
    parser.add_argument("--eval_src_col", default="latin_text")
    parser.add_argument("--eval_ref_col", default="english_text")
    parser.add_argument("--oversample", type=int, default=35)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--min_inter", type=int, default=3)
    parser.add_argument("--compute_comet", action="store_true")

    return parser.parse_args()


def ensure_nltk_resources() -> None:
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)


def main() -> None:
    args = parse_args()
    ensure_nltk_resources()
    if args.mode == "free":
        run_free_mode(args)
    else:
        run_eval_mode(args)


if __name__ == "__main__":
    main()
