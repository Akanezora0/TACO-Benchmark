# GitHub Directory Structure for Reviewers

This document specifies which directories and files should be included in the GitHub repository for reviewers.

## ✅ Directories to Include (Public)

### Core Framework Directories

```
experiments/
├── README.md                           ✅ Main documentation
├── EXPERIMENT_FAIRNESS.md              ✅ Fair comparison principles
├── FINAL_VERSION_SUMMARY.md            ✅ Complete framework summary
├── REVIEWER_GUIDE.md                   ✅ Quick guide for reviewers
├── TACO-SQL实验核心设置与Prompt策略.md    ✅ Detailed prompt strategies
│
├── baselines/                          ✅ Baseline experiments
│   ├── README.md                       ✅ Baseline documentation
│   ├── base_llm/
│   │   ├── README.md                   ✅ Base LLM documentation
│   │   ├── prompt_strategy.py          ✅ Standardized prompt template
│   │   ├── model_wrapper.py            ✅ Unified model interface
│   │   ├── run_experiment.py           ✅ Experiment runner
│   │   ├── batch_evaluate.py           ✅ Batch evaluation
│   │   ├── experiment_config.py        ✅ Model configurations
│   │   └── evaluate_baseline.py        ✅ Baseline evaluation script
│   ├── llm_based/
│   │   ├── README.md                   ✅ LLM-based methods doc
│   │   ├── din_sql/
│   │   │   └── run_din_sql.py          ✅ DIN-SQL runner
│   │   └── mac_sql/
│   │       └── run_mac_sql.py          ✅ MAC-SQL runner
│   ├── sft_based/
│   │   ├── README.md                   ✅ SFT-based methods doc
│   │   ├── codes/
│   │   │   └── run_codes.py            ✅ CodeS runner
│   │   └── qwen_coder/
│   │       └── run_qwen_coder.py       ✅ Qwen2.5-Coder runner
│   └── hybrid/
│       ├── README.md                   ✅ Hybrid methods doc
│       ├── chess/
│       │   └── run_chess.py            ✅ CHESS runner
│       └── zero_nl2sql/
│           └── run_zero_nl2sql.py      ✅ Zero-NL2SQL runner
│
├── taco_sql_exp/                       ✅ TACO-SQL ablation experiments
│   ├── README.md                       ✅ TACO-SQL documentation
│   ├── config.py                       ✅ Experiment configuration
│   ├── experiment_runner.py            ✅ Main experiment runner
│   ├── run_ablation.py                 ✅ Unified ablation script
│   ├── prompts/                        ✅ Prompt strategies
│   │   ├── __init__.py
│   │   ├── question_rewriting_prompt.py  ✅ QR prompt strategy
│   │   ├── query_planning_prompt.py      ✅ QP prompt strategy
│   │   └── sql_generation_prompt.py      ✅ SQL generation prompts
│   ├── utils/                          ✅ Utility functions
│   │   ├── __init__.py
│   │   └── schema_utils.py             ✅ Schema formatting
│   ├── origin/                         ✅ Origin setting
│   │   └── run_origin.py
│   ├── qr/                             ✅ QR setting
│   │   └── run_qr.py
│   ├── qr_tl/                          ✅ QR+TL setting
│   │   └── run_qr_tl.py
│   └── qr_tl_qp/                       ✅ Full TACO-SQL
│       └── run_qr_tl_qp.py
│
└── evaluation/                         ✅ Evaluation framework
    ├── README.md                       ✅ Evaluation documentation
    ├── evaluation_config.py           ✅ Evaluation configuration
    ├── metrics_calculator.py           ✅ Metrics calculation
    ├── error_analysis.py               ✅ Error analysis tools
    ├── exec_eval.py                    ✅ Execution evaluation
    ├── evaluation.py                   ✅ Main evaluation script
    ├── compare.py                      ✅ Result comparison
    ├── draw_result.py                  ✅ Visualization
    └── tex_table.py                    ✅ LaTeX table generation
```

## ❌ Directories to Exclude (Internal)

```
experiments/
├── _internal/                         ❌ Internal development files
│   ├── development_docs/              ❌ Development planning docs
│   ├── logs/                          ❌ Log files
│   ├── temp_analysis/                 ❌ Temporary analysis
│   └── scripts/                       ❌ Internal scripts
│
├── results/                           ❌ Experimental results (optional)
│   ├── raw/                           ❌ Raw results
│   ├── processed/                     ❌ Processed results
│   └── visualizations/                ❌ Visualizations
│
└── scripts/                           ❌ Internal utility scripts
```

## 📋 Summary

### Include in GitHub:
1. ✅ All README.md files (documentation)
2. ✅ All Python scripts (core framework)
3. ✅ All prompt strategy files
4. ✅ Configuration files
5. ✅ Evaluation tools
6. ✅ Fair comparison documentation
7. ✅ Detailed prompt strategy documentation

### Exclude from GitHub:
1. ❌ `_internal/` directory (development files)
2. ❌ `results/` directory (experimental results - optional, can include sample)
3. ❌ Log files (*.log)
4. ❌ Shell scripts (*.sh) - unless essential
5. ❌ Temporary analysis documents
6. ❌ Development planning documents

## 🎯 Key Files for Reviewers

### Must-Have Files:
1. `experiments/README.md` - Main entry point
2. `experiments/TACO-SQL实验核心设置与Prompt策略.md` - Detailed prompt strategies
3. `experiments/EXPERIMENT_FAIRNESS.md` - Fair comparison principles
4. `experiments/REVIEWER_GUIDE.md` - Quick reference
5. `experiments/baselines/base_llm/prompt_strategy.py` - Baseline prompt
6. `experiments/taco_sql_exp/prompts/` - All TACO-SQL prompts

### Recommended Structure for GitHub:

```
experiments/
├── README.md                    # Main documentation
├── EXPERIMENT_FAIRNESS.md       # Fair comparison
├── REVIEWER_GUIDE.md            # Quick guide
├── TACO-SQL实验核心设置与Prompt策略.md  # Detailed prompts
├── baselines/                   # Baseline experiments
├── taco_sql_exp/               # TACO-SQL experiments
└── evaluation/                 # Evaluation tools
```

## 📝 .gitignore Recommendations

Add to `.gitignore`:

```
experiments/_internal/
experiments/results/
experiments/**/*.log
experiments/**/*.sh
experiments/**/temp_*.md
experiments/**/分析*.md
experiments/**/汇总*.md
```

## ✅ Final Checklist

Before publishing to GitHub:

- [ ] Remove `_internal/` directory
- [ ] Remove or clean `results/` directory (keep sample if needed)
- [ ] Remove all `.log` files
- [ ] Remove temporary analysis documents
- [ ] Ensure all README files are in English
- [ ] Ensure all code comments are in English
- [ ] Verify all prompt strategies are documented
- [ ] Check that experimental settings are clear
- [ ] Verify fair comparison principles are documented

