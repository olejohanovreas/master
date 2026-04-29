# Norwegian Sentiment Classification on NoReC

MSc thesis (University of Agder). A controlled comparison on the [Norwegian Review Corpus (NoReC)](https://github.com/ltgoslo/norec) between:

1. Classical baselines (TF-IDF + Logistic Regression / Linear SVM)
2. Fine-tuned NB-BERT-base
3. Zero-shot and few-shot prompting with open-source instruction-tuned LLMs (Llama-3.2-1B / 3B and Llama-3.1-8B)

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
