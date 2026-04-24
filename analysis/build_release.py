"""
PriceBench Release Builder
==========================
Consolidates all experimental results into a clean, public-ready structure.

Notes:
  - IT procurement dataset: n=851 (final cleaned version)
  - Claude Sonnet 4 results use the OFFICIAL Anthropic API (not OpenRouter)
  - All results are 0-shot; 3-shot was exploratory and is not included in release

Output: results_release/
  ├── it_procurement/
  │   └── all_models_0shot.json       ← 7 models × 6 tasks, 0-shot
  ├── cross_domain/
  │   ├── all_models_pred25.json      ← 7 models × 4 domains, PRED25/PRED50/MdAPE
  │   └── popularity_gradient.json   ← 7 models × 4 domains, popular/niche/delta
  ├── ablation/
  │   ├── spec_ablation.json
  │   └── cot_ablation.json
  ├── sensitivity/
  │   └── 3shot_sensitivity.json
  └── README.md
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # project root = PriceBench_Codebase/
RESULTS = BASE / "results"
OUT = BASE / "results_release"

# ── helpers ────────────────────────────────────────────────────────────────

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Saved: {path.relative_to(BASE)}")

# ── 1. IT Procurement: 7 models × 6 tasks, 0-shot ─────────────────────────

def build_it_0shot():
    """Merge consolidated_results_v2_851.json (6 models) + claude files."""

    # Load 6-model base (n=851)
    base = load(RESULTS / "consolidated_results_v2_851.json")

    # Build unified structure
    result = {
        "metadata": {
            "description": "PriceBench IT Procurement — 7 Models × 6 Tasks, 0-shot",
            "n_total": 851,
            "n_task_E": 543,
            "n_task_F_spec_subset": 259,
            "claude_api_note": "Claude Sonnet 4 evaluated via official Anthropic API",
            "models": ["DeepSeek-V3", "Qwen2.5-72B", "GPT-4o-mini",
                       "Mistral Large", "Llama 3.1 70B", "Gemini 2.5 Flash",
                       "Claude Sonnet 4"],
            "tasks": {
                "A": "Direction Classification (Macro-F1)",
                "B": "Reduction Rate Estimation (PRED25)",
                "C": "Fair Price Estimation (PRED25)",
                "D": "Category Classification (Accuracy)",
                "E": "Arithmetic Anomaly Detection (F1, n=543)",
                "F": "Specification Change Detection (Macro-F1, n=259)"
            },
            "prompt_mode": "0-shot",
            "temperature": 0.1
        },
        "results": {}
    }

    # Add 6 base models
    for model, data in base["results"].items():
        shot_data = data.get("0-shot", {})
        result["results"][model] = {}
        for task in ["A", "B", "C", "D", "E", "F"]:
            if task in shot_data:
                result["results"][model][task] = shot_data[task]["metrics"]

    # Add Claude from per-task files
    claude = {}

    # Task A
    ta = load(RESULTS / "claude_it_taskA_20260331_2325.json")
    claude["A"] = {
        "Accuracy": ta["metrics"]["Acc"],
        "Macro-F1": ta["metrics"]["MacroF1"],
        "n": ta["metrics"]["n"]
    }

    # Task B
    tb = load(RESULTS / "claude_it_taskB_20260331_2325.json")
    claude["B"] = {
        "PRED_25": tb["metrics"]["PRED25"],
        "PRED_50": tb["metrics"]["PRED50"],
        "MdAPE_pct": tb["metrics"]["MdAPE"],
        "MAE": tb["metrics"]["MAE"],
        "n": tb["metrics"]["n"]
    }

    # Task C
    tc = load(RESULTS / "claude_it_taskC_20260331_2325.json")
    claude["C"] = {
        "PRED_25": tc["metrics"]["PRED25"],
        "PRED_50": tc["metrics"]["PRED50"],
        "MdAPE_pct": tc["metrics"]["MdAPE"],
        "MAE": tc["metrics"]["MAE"],
        "n": tc["metrics"]["n"]
    }

    # Task D
    td = load(RESULTS / "claude_it_taskD_20260331_2325.json")
    claude["D"] = {
        "Accuracy": td["metrics"]["Acc"],
        "Macro-F1": td["metrics"]["MacroF1"],
        "n": td["metrics"]["n"]
    }

    # Task E
    te = load(RESULTS / "claude_it_taskE_20260401_0857.json")
    claude["E"] = {
        "Precision": te["metrics"]["Prec"],
        "Recall":    te["metrics"]["Rec"],
        "F1":        te["metrics"]["F1"],
        "n":         te["metrics"]["n"]
    }

    # Task F (full dataset: PRED25 inflated; reduced-only is in appendix)
    tf = load(RESULTS / "claude_it_taskF_20260401_0857.json")
    claude["F"] = {
        "PRED_25_all_items":       tf["metrics"]["PRED25"],
        "PRED_50_all_items":       tf["metrics"]["PRED50"],
        "MdAPE_pct_all_items":     tf["metrics"]["MdAPE"],
        "n_all_items":             tf["metrics"]["n"],
        "note": (
            "Task F on full dataset is inflated (63.7% unchanged items). "
            "See reduced_only subset (n=265) for the challenging evaluation."
        )
    }

    result["results"]["Claude Sonnet 4"] = claude

    save(result, OUT / "it_procurement" / "all_models_0shot.json")

# ── 2. [REMOVED] 3-shot not included in release (exploratory early-stage only) ──

# ── 3. Cross-Domain: 7 models × 4 domains ─────────────────────────────────

def build_cross_domain():
    # 6 models from cross_domain_validation
    cross6 = load(RESULTS / "cross_domain_validation_20260331_1544.json")
    # Claude from claude_cross_all
    claude_cross = load(RESULTS / "claude_cross_all_20260401_0857.json")

    DOMAIN_MAP = {
        "consumer_electronics": "Consumer Electronics",
        "used_car":             "Used Cars",
        "luxury_goods":         "Luxury Goods",
        "home_appliance":       "Home Appliances"
    }

    result = {
        "metadata": {
            "description": "PriceBench Cross-Domain Price Estimation — 7 Models × 4 Domains",
            "claude_api_note": "Claude Sonnet 4 evaluated via official Anthropic API",
            "domains": list(DOMAIN_MAP.values()),
            "metric_primary": "PRED(0.25)",
            "metrics_reported": ["PRED25", "PRED50", "MdAPE_pct", "MAE", "n"],
            "models": ["DeepSeek-V3", "Qwen2.5-72B", "GPT-4o-mini",
                       "Mistral Large", "Llama 3.1 70B", "Gemini 2.5 Flash",
                       "Claude Sonnet 4"],
            "note_gemini_used_cars": (
                "Gemini 2.5 Flash Used Cars PRED(0.25)=0.006 is a format-failure "
                "lower bound: 89.1% of responses were non-numeric."
            )
        },
        "results": {}
    }

    # 6 models
    for model, domains in cross6.items():
        result["results"][model] = {}
        for dom_key, dom_label in DOMAIN_MAP.items():
            if dom_key in domains:
                d = domains[dom_key]
                result["results"][model][dom_label] = {
                    "PRED_25": d["PRED25"],
                    "PRED_50": d["PRED50"],
                    "MdAPE_pct": round(d["MdAPE"] * 100, 1) if d["MdAPE"] < 10 else d["MdAPE"],
                    "n": d["valid_n"]
                }

    # Claude
    result["results"]["Claude Sonnet 4"] = {}
    for dom_key, dom_label in DOMAIN_MAP.items():
        if dom_key in claude_cross["cross"]:
            d = claude_cross["cross"][dom_key]
            result["results"]["Claude Sonnet 4"][dom_label] = {
                "PRED_25": d["PRED25"],
                "PRED_50": d["PRED50"],
                "MdAPE_pct": d["MdAPE"],
                "n": d["n"]
            }

    save(result, OUT / "cross_domain" / "all_models_pred25.json")

# ── 4. Popularity Gradient: 7 models × 4 domains ─────────────────────────

def build_popularity():
    # 6 models
    pop6 = load(RESULTS / "popularity_gradient_20260331_1612.json")
    # Claude
    claude_cross = load(RESULTS / "claude_cross_all_20260401_0857.json")

    DOMAIN_MAP = {
        "consumer_electronics": "Consumer Electronics",
        "used_car":             "Used Cars",
        "luxury_goods":         "Luxury Goods",
        "home_appliance":       "Home Appliances"
    }

    result = {
        "metadata": {
            "description": "PriceBench Popularity Gradient — 7 Models × 4 Domains",
            "explanation": (
                "Positive delta = popular items easier to price (memorization advantage). "
                "Negative delta = luxury reversal (aspirational price inflation in training data)."
            ),
            "popular_criteria": [
                "Top-50 organic search rank on major Chinese e-commerce platforms",
                "Baidu Index daily average >= 500",
                "At least 1,000 product reviews on major platforms"
            ]
        },
        "results": {}
    }

    for model, domains in pop6.items():
        if model in ("Claude Sonnet 3.5",):
            continue  # skip failed runs
        result["results"][model] = {}
        for dom_key, dom_label in DOMAIN_MAP.items():
            if dom_key in domains:
                d = domains[dom_key]
                result["results"][model][dom_label] = {
                    "popular_PRED25": d["popular"]["PRED25"],
                    "popular_n":      d["popular"]["valid_n"],
                    "niche_PRED25":   d["niche"]["PRED25"],
                    "niche_n":        d["niche"]["valid_n"],
                    "delta":          d["gap"]
                }

    # Claude from claude_cross_all popularity section
    result["results"]["Claude Sonnet 4"] = {}
    for dom_key, dom_label in DOMAIN_MAP.items():
        if dom_key in claude_cross["popularity"]:
            d = claude_cross["popularity"][dom_key]
            result["results"]["Claude Sonnet 4"][dom_label] = {
                "popular_PRED25": d["popular_p25"],
                "popular_n":      d["popular_n"],
                "niche_PRED25":   d["niche_p25"],
                "niche_n":        d["niche_n"],
                "delta":          d["delta"]
            }

    save(result, OUT / "cross_domain" / "popularity_gradient.json")

# ── 5. Ablation ────────────────────────────────────────────────────────────

def build_ablation():
    ab = load(RESULTS / "ablation_results.json")

    # Spec ablation: extract cleanly
    spec = {
        "metadata": {
            "description": "Specification Ablation — Task C & B, IT Procurement (n=259, items with spec text)",
            "note": "Subset of 259 items that have non-empty specification text."
        },
        "task_C": {},
        "task_B": {}
    }

    cot = {
        "metadata": {
            "description": "Chain-of-Thought Ablation — Task C, IT Procurement",
            "prompt_variant": "Appended 'Let's think step by step.' to Task C prompt"
        },
        "results": {}
    }

    # Parse ablation_results.json — structure: {spec_ablation: {model: {task: {with_spec, no_spec}}}}
    if "spec_ablation" in ab:
        for model, tasks in ab["spec_ablation"].items():
            if "C" in tasks:
                tc = tasks["C"]
                spec["task_C"][model] = {
                    "with_spec_PRED25":    tc["with_spec"].get("PRED_25"),
                    "without_spec_PRED25": tc["no_spec"].get("PRED_25"),
                    "delta": round(
                        tc["with_spec"].get("PRED_25", 0) -
                        tc["no_spec"].get("PRED_25", 0), 4)
                }
            if "B" in tasks:
                tb = tasks["B"]
                spec["task_B"][model] = {
                    "with_spec_PRED25":    tb["with_spec"].get("PRED_25"),
                    "without_spec_PRED25": tb["no_spec"].get("PRED_25"),
                    "delta": round(
                        tb["with_spec"].get("PRED_25", 0) -
                        tb["no_spec"].get("PRED_25", 0), 4)
                }

    if "cot_ablation" in ab:
        for model, data in ab["cot_ablation"].items():
            cot["results"][model] = {
                "direct_PRED25": data.get("direct", {}).get("PRED_25"),
                "cot_PRED25":    data.get("cot",    {}).get("PRED_25"),
                "delta": round(
                    (data.get("cot", {}).get("PRED_25", 0) or 0) -
                    (data.get("direct", {}).get("PRED_25", 0) or 0), 4)
            }

    save(spec, OUT / "ablation" / "spec_ablation.json")
    save(cot,  OUT / "ablation" / "cot_ablation.json")

# ── 6. 3-Shot Sensitivity ──────────────────────────────────────────────────

def build_sensitivity():
    s = load(RESULTS / "3shot_sensitivity_20260331_0415.json")

    result = {
        "metadata": {
            "description": "3-Shot Sensitivity Analysis — Gemini 2.5 Flash × Tasks C/D/E × 3 seeds",
            "seeds": [42, 123, 456],
            "finding": (
                "Near-zero variance (CV < 3.3%) confirms 3-shot degradation is "
                "systematic, not seed-dependent."
            )
        },
        "raw": s
    }

    save(result, OUT / "sensitivity" / "3shot_sensitivity.json")

# ── 7. README ──────────────────────────────────────────────────────────────

README = """\
# PriceBench — Experimental Results (Open Release)

This directory contains the complete experimental results for the paper:

> **PriceBench: Do LLMs Understand Prices, or Just Memorize Them?**  
> NeurIPS 2026 Datasets & Benchmarks Track (under review)

## Directory Structure

```
results_release/
├── it_procurement/
│   └── all_models_0shot.json       # 7 models × 6 tasks, 0-shot (PRIMARY)
├── cross_domain/
│   ├── all_models_pred25.json      # 7 models × 4 consumer domains
│   └── popularity_gradient.json   # Popular vs. Niche breakdown
├── ablation/
│   ├── spec_ablation.json         # With/without specification text
│   └── cot_ablation.json          # CoT vs. direct prompting
└── sensitivity/
    └── 3shot_sensitivity.json     # 3-seed stability test (Gemini)
```

## Models Evaluated

| Model | Provider | Type | Parameters |
|-------|----------|------|-----------|
| DeepSeek-V3 | deepseek-chat | Open | ~670B (MoE) |
| Qwen2.5-72B | qwen-plus | Open | 72B |
| GPT-4o-mini | OpenAI | Closed | — |
| Mistral Large | MistralAI | Closed | ~123B |
| Llama 3.1 70B | Together AI | Open | 70B |
| Gemini 2.5 Flash | Google AI | Closed | — |
| Claude Sonnet 4 | Anthropic (official API) | Closed | — |

## Key Metrics

- **PRED(0.25)**: Fraction of predictions within 25% of true price (primary)  
- **PRED(0.50)**: Fraction within 50% (secondary)  
- **MdAPE**: Median Absolute Percentage Error  
- **Macro-F1**: For classification tasks  

## Notes

- All IT procurement results use **n=851** (final cleaned dataset)  
- Task E uses n=543 (items with unit_price × quantity available)  
- Task F (reduced-only) uses n=265 (items with >5% expert reduction)  
- Gemini 2.5 Flash Used Cars: PRED(0.25)=0.006 is a format-failure lower bound (89.1% non-numeric)  
- **Claude Sonnet 4** was evaluated via the **official Anthropic API** (not proxy services)  
- All evaluations are 0-shot; the 3-shot sensitivity file is a targeted stability study for Gemini only  

## Citation

```bibtex
@inproceedings{pricebench2026,
  title     = {PriceBench: Do LLMs Understand Prices, or Just Memorize Them?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Datasets and Benchmarks Track}
}
```
"""

def build_readme():
    path = OUT / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(README, encoding="utf-8")
    print(f"  ✅ Saved: results_release/README.md")

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building PriceBench Release Package...")
    print()
    print("1. IT Procurement 0-shot (7 models, n=851)...")
    build_it_0shot()
    print("2. Cross-Domain results (7 models)...")
    build_cross_domain()
    print("3. Popularity Gradient (7 models)...")
    build_popularity()
    print("4. Ablation studies...")
    build_ablation()
    print("5. 3-Shot Sensitivity...")
    build_sensitivity()
    print("6. README...")
    build_readme()
    print()
    print("✅ Done! Output at: results_release/")

