# Handover notes for a future Claude session

This is the continuity document for a future Claude chat that picks up this thesis project. It assumes nothing about session memory persistence and tries to give you everything you need to be useful within five minutes of opening this file.

## What this is

MSc thesis at the University of Agder (UiA), Faculty of Engineering and Science, Department of Information and Communication Technology. Author: **Ole-Johan Øvreås**. Topic: a controlled empirical comparison on the Norwegian Review Corpus (NoReC) between (i) classical TF-IDF baselines, (ii) fine-tuned NB-BERT-base, and (iii) prompted Llama 3 Instruct (1B / 3B / 8B), in zero-shot and four-shot regimes.

## Where the thesis stands right now

**The main draft is complete.** ~34 body pages, ~48 PDF pages including front and back matter. Headline result is robust under multi-seed variance (~6 pooled std deviations between best fine-tune and best LLM); ablations across prompt wording, demo content, demo count, and demo order all bound the LLM-vs-fine-tune lead well within the gap.

The user's posture is "good enough to pass" — they have a full-time job and want a defensible thesis without chasing high marks. Don't propose extra experiments unless asked.

## Two repositories

**Code repo** (this directory): `~/Code/master/`, GitHub remote `git@github.com:olejohanovreas/master-new.git`. Branch `main`. As of this writeup, `main` is roughly 14+ commits ahead of `origin/main`; `git push` when the user asks.

**Thesis text** (LaTeX in Overleaf): mirrored locally at `~/overleaf-mirror/69f1e32653de3bed2edb2172/`. This is a separate filesystem; the thesis text is **not** in the code repo. Sync to/from Overleaf is via `overleaf-cli start` running in a terminal owned by the user. See "Overleaf gotchas" below.

There used to be a predecessor GitHub repo at `olejohanovreas/master`. We purged the V100 working tree from it. The user can delete the GitHub-side repo via the web UI when convenient.

## What runs where

**Local (RTX 3070, 8 GB VRAM)**: classical baselines, all writeup analysis, occasional LLM smoke tests. Light work only.

**UiA V100** (Tesla V100-SXM3, 32 GB VRAM, dedicated, no use limits): everything heavy. Reached over eduVPN through the Coder workspace platform; SSH alias is `coder.master`. The `coder` CLI's `Host coder.*` block in `~/.ssh/config` makes this work without the `coder connect` daemon. There is a benign coder client/server version-mismatch warning prefixed to every SSH call — ignore it.

V100 working directory: `~/master-new/` on the remote. Uses an independent uv venv that shares `uv.lock` with local.

## Operational gotchas (each cost time when first encountered)

**The cu126 PyTorch pin.** The V100 is compute capability 7.0 (Volta). Recent default PyTorch wheels (cu128, cu130) dropped Volta from their build matrix; `torch.cuda.is_available()` returns `True` but every kernel launch raises `no kernel image is available`. This silently fell back to CPU once during NB-BERT fine-tuning before being diagnosed (135× slower than expected). Fixed via `[tool.uv.sources] torch = { index = "pytorch-cu126" }` in `pyproject.toml`. Don't change this pin without testing both machines.

**Overleaf-cli sync flakiness.** The daemon (`overleaf-cli start`, runs in a user-owned terminal) is the only sync mechanism. It has three known bad behaviours we've hit:
1. Some text edits don't propagate up. Workaround: append a trailing newline to nudge the watcher.
2. Newly-created files (new chapters, new figures) sometimes fail to sync because the cloud-side file tree extension cache is stale. Workaround: refresh the Overleaf web UI tab (F5), or for binaries, drag-drop into the Overleaf web UI directly. The error to look for in the daemon log is `File not found in file tree: <path>`.
3. Binary files (PNGs, PDFs) get **zeroed out** by the sync agent. New figures should be uploaded directly via the Overleaf web UI's drag-drop, not relied upon to sync from the local mirror.
4. On daemon restart, the agent does a fresh scan and can pull stale cloud content over local edits if the cloud was out of date for any reason. Verify local files survived after a restart.

These are all daemon bugs, not anything we can fix in our scripts.

**Bash `set -e` doesn't catch failures across `| tail -5`.** The Tier B orchestrator originally piped each python invocation through `tail -5` to keep the orchestrator log compact. This silently masked OOM failures from `set -e`, so k=6 and k=8 ran for ~5 minutes each, OOM'd, and the orchestrator advanced to the next iteration with no error visible. Always check that expected output files actually exist before assuming a sweep completed.

**8B prompts can exceed `MAX_INPUT_TOKENS=4096` at high k.** k=8 (8 demos × ~150 tokens + 1500-token review + chat overhead) sometimes pushes individual inputs past 4096 tokens. The chat template's trailing assistant generation prompt then gets right-truncated, and the model produces continuations of the user's review text instead of a verdict. This produced 500 unparseable responses at k=8 in our run. The fix would be to raise the cap; we documented the failure rather than fixing it because it's an interesting finding for the discussion. Don't be surprised by this if you re-run with longer prompts.

**HF token.** A personal access token used during this work has since been revoked at huggingface.co/settings/tokens. Any new run needs a fresh token via `uv run hf auth login`.

## How to find things

**Headline result table**: `results/summary_table.md` (also `.csv`).

**Per-experiment metrics**: `results/{classical,nbbert,llm__s*_p*,nbbert_chunked}.json`.

**Per-config predictions**: `results/{classical,llm,nbbert}_preds*.csv` (gitignored — large).

**Figures used in the thesis**: `results/figures/*.png`. Mirror in `~/overleaf-mirror/.../Figures/results/` — be aware the Overleaf-mirror copies are unreliable due to the binary-corruption bug; when in doubt, drag-drop from the local repo into the Overleaf web UI.

**Thesis chapters**: `~/overleaf-mirror/69f1e32653de3bed2edb2172/chapters/{intro,theory,method,implementation,results,discussion,conclusion,abstract,acknowledgements,declaration}.tex`, plus `appendices/appendix.tex`. Chapter inputs in `main.tex`.

**Bibliography**: `~/overleaf-mirror/69f1e32653de3bed2edb2172/bibliography.bib`. Uses biblatex.

**Memory directory**: `~/.claude/projects/-home-olejohan-Code-master/memory/`. Contains user profile, working preferences, project decisions, etc. Updates persist across Claude sessions.

## User working preferences (from memory; restated for convenience)

- Pragmatic "good enough to pass" mode. Don't propose extra rigor or experiments unless asked.
- Commit at the end of each project phase without asking.
- When the user says "delete" or "purge", do it fully and immediately. No hedging or proposing soft restoration.
- Minimise supervisor interaction until the draft is ready. Don't suggest looping in the supervisor for design decisions.
- Pragmatic rigor: the user has been willing to add multi-seed runs, ablations, validity sections — but only when proposed as concrete high-ROI items with clear page-count and compute estimates.

## What's left to do

The final-polish pass has been done. Supervisor name (Turgay Celik) is in `titlepage.tex`; the UiA declaration form has been removed from the build (`chapters/declaration.tex` deleted, `\input` line removed from `main.tex`); a typo / consistency read-through has been done across all chapters (numerical errors fixed, British-English spelling normalised for *artefact* and *tokeniser*, a few awkward phrasings cleaned up); the `per_source_heatmap.png` and `per_year.png` figures have been uploaded directly via the Overleaf web UI; the HF token has been revoked; the predecessor GitHub repo has been deleted; the local commits have been pushed to `origin/main`.

There is no outstanding blocker. The remaining items are user-side at submission: filling in any university paperwork outside the thesis PDF itself.

## Possible future-work items the user might raise

If they want more pages or more rigor, here's what's plausible (in approximate order of ROI):

- **Long-context Norwegian encoder** (e.g. NorBERT-3, if available) as a control for the long-review penalty. Currently the chunk-and-pool ablation tests this with NB-BERT-base only.
- **Other LLM families**: Mistral, Qwen, Gemma — would test how much of the prompted-LLM advantage is family-specific vs scale-driven.
- **Demonstration content variation beyond the within-Llama-3 multi-seed**: e.g. handpicked vs random, varied length.
- **More seeds**: 5 → 10 would tighten the confidence intervals further.
- **Encoder hyperparameter search**: NB-BERT was fine-tuned with standard defaults rather than a per-dataset sweep. A small search might extract another 0.5-1pp.
- **Ensemble of all three approaches**: Section 6.4 of the Discussion notes that BERT and LLM make different mistakes; a majority-vote ensemble might genuinely help. Real but modest extra work.

## Snapshot of commits at handover time

```
9124512  Tier B integration: k-shot and demo-order ablations on Llama-3.1-8B
985bd21  Tier-B prep + per-source/year analysis + classical hyperparam capture
45fda8b  Phases 8-10: multi-seed runs, chunk-and-pool, prompt sensitivity
812750f  Phase 7 prep: multi-seed, chunk-and-pool, and prompt variants
7e53f38  Phase 6: results aggregation and error analysis
02fdc33  README: switch LLM family description back to Llama 3
a67781d  Phase 5: zero-shot and few-shot prompting eval (Llama 3 family)
553d372  Phase 4: NB-BERT-base fine-tune
6a74ade  Phase 4 prep: V100 remote workflow
4b8b34e  Phase 3: classical TF-IDF baselines (LR + Linear SVM)
0308876  Phase 2: NoReC loader and dataset statistics
d851da5  Phase 1: project skeleton and uv-managed Python environment
e5ae9e2  first commit
```

## When in doubt

Read `results/summary_table.md` for the canonical result picture. Read the most recent commit messages for what changed last. Check `~/.claude/projects/-home-olejohan-Code-master/memory/MEMORY.md` for the user-pref index. Trust the local mirror over the Overleaf web UI when they disagree.
