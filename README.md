# Norwegian Sentiment Classification on NoReC

MSc thesis (University of Agder, AI). A controlled empirical comparison on the [Norwegian Review Corpus (NoReC)](https://github.com/ltgoslo/norec) between three approaches to document-level binary sentiment classification:

1. **Classical baselines** — TF-IDF + Logistic Regression / Linear SVM
2. **Fine-tuned NB-BERT-base** — Norwegian monolingual encoder fine-tune
3. **Prompted Llama 3 Instruct** — zero-shot and four-shot, three sizes (1B / 3B / 8B)

Task: document-level binary sentiment with the canonical label mapping `{1,2,3} → negative`, `{4,5,6} → positive`. NoReC's published train/dev/test splits.

## Headline result

| configuration | accuracy | macro-F1 |
|---|---|---|
| majority floor | 0.7753 | 0.4367 |
| Logistic Regression (balanced) | 0.8765 | 0.8190 |
| NB-BERT-base, 5 seeds | 0.8994 ± 0.0033 | 0.8529 ± 0.0038 |
| NB-BERT-base + chunk-and-pool, 5 seeds | 0.9054 ± 0.0020 | 0.8623 ± 0.0022 |
| **Llama-3.1-8B 4-shot, 5 seeds** | **0.9131 ± 0.0044** | **0.8777 ± 0.0036** |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
git clone --depth 1 https://github.com/ltgoslo/norec.git data/norec
```

PyTorch is pinned to the **cu126** wheel index. More recent CUDA-13 wheels drop Volta (compute capability 7.0), which the V100 used here has — without the pin, every GPU kernel launch fails. See `pyproject.toml`.

Llama models are gated on the Hugging Face Hub. Authenticate once with `uv run hf auth login` before running any LLM script.

## Layout

- `thesis/` — importable package: data loader, classical / transformer / LLM pipelines, metrics
- `scripts/` — entry-point runners and analysis. `v100.sh` wraps the remote workflow on the UiA V100
- `results/` — per-experiment metrics JSONs, aggregate tables, and figures used in the thesis
- `data/`, `checkpoints/` — gitignored, regenerable

Seeds are CLI flags on every script (default 42). Multi-seed configurations are run across {42, 43, 44, 45, 46}.

## License

Code is for academic use. NoReC is distributed by Språkbanken / University of Oslo under CC BY-NC 4.0.
