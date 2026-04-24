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
| Claude Sonnet 4 | Anthropic | Closed | — |

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
- All evaluations are 0-shot; the 3-shot sensitivity file is a targeted stability study for Gemini only  
- **Task F** is Spec Change Prediction (binary, Macro-F1 on n=259 spec-text subset); most models achieve baseline F1≈0.163  

## Data Provenance

| Source | Description |
|--------|-------------|
| `consolidated_results_v2_851.json` | 0-shot results for 6 models (n=851) |
| `claude_it_task[A-F]_*.json` | Claude Sonnet 4 IT results (official Anthropic API) |
| `cross_domain_validation_20260331_1544.json` | 6-model cross-domain results |
| `claude_cross_all_20260401_0857.json` | Claude cross-domain + popularity (official API) |
| `popularity_gradient_20260331_1612.json` | 6-model popularity gradient |
| `ablation_results.json` | Spec & CoT ablation |
| `3shot_sensitivity_20260331_0415.json` | 3-seed stability test |

## Citation


```bibtex
@inproceedings{pricebench2026,
  title     = {PriceBench: Do LLMs Understand Prices, or Just Memorize Them?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Datasets and Benchmarks Track}
}
```
