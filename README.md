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
| NB-BERT-base + chunk-and-pool | 0.9030 | 0.8544 |
| **Llama-3.1-8B 4-shot, 5 seeds** | **0.9131 ± 0.0044** | **0.8777 ± 0.0036** |

The 8B prompted LLM with four in-context demonstrations beats the fine-tuned encoder by roughly six pooled standard deviations on macro-F1 — a real, not-seed-noise gap. Within the Llama family, seed variance scales inversely with size (1B ±0.043, 3B ±0.017, 8B ±0.0036). The encoder's penalty on long reviews is *architectural*, not a truncation artefact (chunk-and-pool does not close the gap).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
```

This installs all dependencies pinned in `uv.lock`, including PyTorch from the **cu126** wheel index. The cu126 pin matters: more recent CUDA-13 PyTorch wheels drop Volta (compute capability 7.0), which the V100 used for this project has. Without the pin, `torch.cuda.is_available()` returns `True` but every GPU kernel launch fails with `no kernel image is available for execution on the device`. See `pyproject.toml` for the exact source override.

## Repository layout

```
master/
├── pyproject.toml          uv-managed deps + Python pin
├── uv.lock                 exact pins for every transitive dep
├── README.md               this file
├── handover.md             continuity context for a future session
├── thesis/                 importable package
│   ├── data.py             NoReC loader + binary label mapping
│   ├── classical.py        TF-IDF + LR / LinearSVC pipelines
│   ├── transformer.py      NB-BERT tokenization + chunk-and-pool inference
│   ├── llm.py              prompt assembly, parser, demo selection / shuffling
│   └── evaluation.py       accuracy, macro-F1, per-class metrics
├── scripts/                entry-point runners + analysis
│   ├── data_stats.py       NoReC distribution sanity check
│   ├── run_classical.py    TF-IDF sweep over (clf, balanced, C)
│   ├── run_finetune_nbbert.py    fine-tune NB-BERT-base, --seed configurable
│   ├── eval_nbbert_chunked.py    chunk-and-pool re-evaluation
│   ├── run_llm.py          zero/few-shot LLM eval, --seed / --prompt /
│   │                       --n_per_class / --demo_order_seed
│   ├── aggregate_metrics.py     unified mean ± std summary table
│   ├── error_analysis.py   confusion / per-category / per-length / qual samples
│   ├── per_source_year.py  per-NoReC-source and per-year breakdowns
│   └── v100.sh             V100 push/pull/run helper
├── data/                   NoReC clone (gitignored, populated by bootstrap)
├── results/                metrics + figures (per-config CSVs gitignored)
│   └── figures/            PNGs used in the thesis
├── checkpoints/            saved fine-tunes (gitignored)
└── papers/                 background literature (PDFs)
```

## Running everything from a clean checkout

The full experimental sequence, in order:

```bash
# 1. environment + data
uv sync
git clone --depth 1 https://github.com/ltgoslo/norec.git data/norec

# 2. classical baselines (CPU, ~5 min)
uv run python scripts/run_classical.py

# 3. NB-BERT × 5 seeds (V100, ~80 min total)
for s in 42 43 44 45 46; do
  uv run python scripts/run_finetune_nbbert.py --seed $s
done

# 4. NB-BERT chunk-and-pool re-eval (V100, ~5 min)
uv run python scripts/eval_nbbert_chunked.py

# 5. LLM zero+few-shot, all sizes, multi-seed (V100, several hours)
for s in 42 43 44 45 46; do
  uv run python scripts/run_llm.py --seed $s --regimes few-shot \
    --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct
  uv run python scripts/run_llm.py --seed $s --regimes few-shot \
    --models meta-llama/Llama-3.1-8B-Instruct --batch_size 4
done
# zero-shot is deterministic, run once at seed 42:
uv run python scripts/run_llm.py --seed 42 --regimes zero-shot \
  --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
           meta-llama/Llama-3.1-8B-Instruct

# 6. ablations (V100, ~3 hours)
for p in terse norwegian; do
  uv run python scripts/run_llm.py --seed 42 --prompt $p --regimes few-shot \
    --models meta-llama/Llama-3.1-8B-Instruct --batch_size 4
done
for n in 1 3; do  # k=2 and k=6 fit at batch 4
  uv run python scripts/run_llm.py --seed 42 --regimes few-shot \
    --models meta-llama/Llama-3.1-8B-Instruct --batch_size 4 --n_per_class $n
done
uv run python scripts/run_llm.py --seed 42 --regimes few-shot \
  --models meta-llama/Llama-3.1-8B-Instruct --batch_size 2 --n_per_class 4  # k=8 needs batch 2
for o in 1 2 3; do
  uv run python scripts/run_llm.py --seed 42 --regimes few-shot \
    --models meta-llama/Llama-3.1-8B-Instruct --batch_size 4 --demo_order_seed $o
done

# 7. analysis (CPU, seconds)
uv run python scripts/aggregate_metrics.py
uv run python scripts/error_analysis.py
uv run python scripts/per_source_year.py
```

Llama models are gated on the Hugging Face Hub. Authenticate once with `uv run hf auth login` and a personal token before running any of the LLM scripts.

## Running on the UiA V100

Local development happens on consumer hardware; GPU work is offloaded to a Tesla V100 hosted in the University of Agder's Coder workspace. Connection requires eduVPN plus the `coder` CLI's SSH config (alias `coder.master`). The `scripts/v100.sh` helper wraps the workflow:

```bash
./scripts/v100.sh bootstrap          # one-time: install uv, sync, clone NoReC on remote
./scripts/v100.sh push               # rsync local code → V100
./scripts/v100.sh run scripts/foo.py # push + run on V100 + pull results
./scripts/v100.sh pull               # pull results/ from V100
./scripts/v100.sh ssh [args]         # open remote shell or run a remote command
```

Local `.venv` and remote `~/master-new/.venv` are independent but share `uv.lock`, so package versions agree.

## Reproducibility notes

- Seeds are passed as CLI arguments to every script. Default 42. NB-BERT and LLM few-shot configurations are run across seeds {42, 43, 44, 45, 46} and reported as mean ± std. Classical baselines and LLM zero-shot are deterministic.
- All package versions pinned via `uv.lock`. Re-syncing on any machine produces a bit-identical environment.
- The cu126 PyTorch pin is the single most important reproducibility detail (see Setup).
- Per-configuration prediction CSVs and metrics JSONs use a deterministic naming scheme so results aggregation is order-independent: re-running just the analysis stage after adding a new configuration takes seconds.

## Outputs

Tracked in git (small):
- `results/summary_table.{csv,md}` — unified mean ± std table across all configurations
- `results/per_category.csv`, `results/per_length.csv`, `results/per_source.csv`, `results/per_year.csv`
- `results/error_samples.md` — qualitative misclassification examples
- `results/figures/*.png` — figures used in the thesis
- `results/{classical,nbbert,llm,nbbert_chunked,llm__s42_p*}.json` — per-experiment metrics

Gitignored (large, regenerable):
- `results/*_preds*.csv` — per-configuration test predictions
- `data/`, `checkpoints/`, `.venv/`

## Thesis text

Written in LaTeX in a separate Overleaf project, mirrored locally at `~/overleaf-mirror/69f1e32653de3bed2edb2172/`. The thesis cites results, tables, and figures produced by this repository.

## License

Code is for academic use. NoReC is distributed by Språkbanken / University of Oslo under CC BY-NC 4.0; see [`data/norec/`](https://github.com/ltgoslo/norec) for terms.
