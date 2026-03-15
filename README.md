# latin-rag-translator

Production-ready CLI package for **Latin → English** translation using:

- Chroma retrieval over parallel Latin data
- NLLB for draft translation
- Qwen chat model for final refinement

This repository is structured so a user can install and run with a single command once models are available on Hugging Face.

## 1) Publish required assets on Hugging Face

You said you will host trained models and Chroma artifacts on Hugging Face. Use this mapping:

- `NLLB checkpoint` → model repo (example: `your-org/nllb-latin-en`)
- `Qwen LoRA adapter` (optional) → model repo (example: `your-org/qwen-latin-refine-lora`)
- `Chroma persist directory` → dataset repo with a zip/tarball (example: `your-org/latin-chroma-db`)

## 2) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

After this, the CLI is available as:

```bash
latin-rag-translator --help
```

## 3) Quick start (free translation mode)

First download/unpack your Chroma directory locally (for example into `./data/chroma_db`).

```bash
latin-rag-translator \
  --mode free \
  --persist_dir ./data/chroma_db \
  --nllb_model your-org/nllb-latin-en \
  --qwen_model Qwen/Qwen1.5-32B-Chat \
  --qwen_lora_path ./models/qwen-lora \
  --text "Gallia est omnis divisa in partes tres."
```

You can also translate multiple lines from a file:

```bash
latin-rag-translator \
  --mode free \
  --persist_dir ./data/chroma_db \
  --nllb_model your-org/nllb-latin-en \
  --qwen_model Qwen/Qwen1.5-32B-Chat \
  --input_file latin.txt \
  --output_file predictions.jsonl
```

## 4) Evaluation mode

```bash
latin-rag-translator \
  --mode eval \
  --persist_dir ./data/chroma_db \
  --nllb_model your-org/nllb-latin-en \
  --qwen_model Qwen/Qwen1.5-32B-Chat \
  --max_examples 100 \
  --output_file eval_outputs.jsonl
```

Optional COMET:

```bash
latin-rag-translator ... --mode eval --compute_comet
```

## 5) Recommended repository publishing workflow

1. Create a new GitHub repository (e.g., `latin-rag-translator`).
2. Push this code.
3. Add release notes with exact model IDs and Chroma download command.
4. Add a `config.example.env` if you want users to copy/paste model IDs.

## Notes

- GPU is strongly recommended for Qwen/NLLB inference.
- `--no_4bit` disables 4-bit quantized loading for Qwen.
- Chroma retrieval quality can be tuned with `--top_k`, `--oversample`, `--alpha`, `--beta`, and `--min_inter`.
