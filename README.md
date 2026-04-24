# PriceBench Evaluation Codebase

This repository contains the benchmark dataset, reproducible evaluation pipeline, and experimental results for the PriceBench paper (NeurIPS 2026 Datasets & Benchmarks Track submission).

> **PriceBench: Do LLMs Understand Prices, or Just Memorize Them?**

## Repository Structure

```
PriceBench_Codebase/
├── dataset/                            ← PriceBench benchmark data
│   ├── it_procurement/
│   │   └── bench_v2.json              # IT Procurement benchmark (n=851 items, 6 tasks A–F)
│   └── cross_domain/
│       ├── electronics_raw.json       # Consumer Electronics (popular + niche, n≈163)
│       ├── used_cars_raw.json         # Used Cars (popular + niche, n≈156)
│       ├── luxury_goods_raw.json      # Luxury Goods (popular + niche, n≈168)
│       └── appliances_raw.json        # Home Appliances (popular + niche, n≈164)
├── results_release/                    ← Pre-computed experimental results (all 7 models)
│   ├── it_procurement/
│   │   └── all_models_0shot.json      # 7 models × 6 tasks, 0-shot (primary results)
│   ├── cross_domain/
│   │   ├── all_models_pred25.json     # 7 models × 4 domains, PRED(0.25/0.50)/MdAPE
│   │   └── popularity_gradient.json   # Popular vs. Niche breakdown per model × domain
│   ├── ablation/
│   │   ├── spec_ablation.json         # With vs. without specification text (Task C & B)
│   │   └── cot_ablation.json          # Chain-of-Thought vs. direct prompting
│   └── sensitivity/
│       └── 3shot_sensitivity.json     # 3-seed 3-shot stability study (Gemini 2.5 Flash)
├── evaluation/
│   ├── 05_llm_eval_async.py           # High-concurrency evaluation core (IT Procurement)
│   └── 14_cross_domain_eval.py        # Cross-domain consumer price evaluation engine
├── data_preparation/
│   └── 01_filter_data.py              # Script for building the 851-item final dataset
├── analysis/
│   └── build_release.py               # Post-processes raw LLM inferences into results_release/
├── requirements.txt
└── README.md
```

## Setup & Environment

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure API Keys:**
    API keys are loaded from environment variables:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="AIza..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENROUTER_KEY="sk-or-..."   # optional, for multi-model testing
    ```

## Execution

### Run Full IT Procurement Evaluation
```bash
python3 evaluation/05_llm_eval_async.py
```
- Adjust concurrency: `--concurrency 15`
- Target specific models: `python3 05_llm_eval_async.py gpt-4o-mini claude-sonnet-official`

### Run Cross-Domain Evaluation
```bash
python3 evaluation/14_cross_domain_eval.py
```

### Rebuild Results Release Package
After raw evaluation logs are written, recompute the unified `results_release/` JSON arrays:
```bash
python3 analysis/build_release.py
```

## Data

The benchmark dataset is included directly in this repository under `dataset/`:

- **IT Procurement** (`dataset/it_procurement/bench_v2.json`): 851 real-world software procurement price review records with expert-annotated ground-truth prices, covering 6 structured tasks (A–F).
- **Cross-Domain Consumer** (`dataset/cross_domain/`): 651 consumer items across 4 domains, each labeled with a popularity gradient (popular / niche) based on web-corpus coverage.

Pre-computed model outputs for all 7 evaluated LLMs are in `results_release/` for direct inspection without re-running the API calls.
