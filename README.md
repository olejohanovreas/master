# Norwegian Sentiment Classification on NoReC

MSc thesis (University of Agder). A controlled comparison on the [Norwegian Review Corpus (NoReC)](https://github.com/ltgoslo/norec) between:

1. Classical baselines (TF-IDF + Logistic Regression / Linear SVM)
2. Fine-tuned NB-BERT-base
3. Zero-shot and few-shot prompting with open-source instruction-tuned LLMs (Llama 3 Instruct: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B)

Task: document-level **binary** sentiment classification with label mapping `{1,2,3} → negative`, `{4,5,6} → positive`. Splits: NoReC's published train/dev/test.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
```

## Layout

- `thesis/` — package with reusable modules (data, models, evaluation)
- `scripts/` — entry-point experiment scripts
- `configs/` — experiment configuration files
- `data/` — local data cache (gitignored)
- `results/` — experiment outputs (gitignored)
- `papers/` — background literature

## Running on the UiA V100

GPU work runs on the UiA Coder V100. Connection requires eduVPN + the `coder` CLI's SSH config (`coder.master`). Use the `scripts/v100.sh` helper:

```bash
./scripts/v100.sh bootstrap          # one-time: install uv, sync deps, clone NoReC
./scripts/v100.sh push               # sync local code to V100
./scripts/v100.sh run scripts/foo.py # push + run + pull results
./scripts/v100.sh ssh                # open shell on V100
```

Local `.venv` and remote `~/master-new/.venv` are independent but share the same `uv.lock`, so package versions match.
