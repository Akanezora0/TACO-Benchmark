# GitHub Publication Guide

This guide specifies exactly which directories and files should be published to GitHub for reviewers.

## ✅ Directories to Publish (Public Repository)

### 1. Core Documentation (Root Level)

```
experiments/
├── README.md                           ✅ Main entry point
├── EXPERIMENT_FAIRNESS.md              ✅ Fair comparison principles
├── FINAL_VERSION_SUMMARY.md            ✅ Complete framework summary
├── REVIEWER_GUIDE.md                   ✅ Quick reviewer guide
├── TACO-SQL实验核心设置与Prompt策略.md    ✅ Detailed prompt strategies
└── GITHUB_DIRECTORY_STRUCTURE.md       ✅ This guide (optional)
```

### 2. Baseline Experiments

```
experiments/baselines/
├── README.md                           ✅ Baseline documentation
├── base_llm/
│   ├── README.md                       ✅ Base LLM documentation
│   ├── prompt_strategy.py              ✅ Standardized prompt template
│   ├── model_wrapper.py                ✅ Unified model interface
│   ├── run_experiment.py               ✅ Experiment runner
│   ├── batch_evaluate.py               ✅ Batch evaluation
│   ├── experiment_config.py            ✅ Model configurations
│   └── evaluate_baseline.py            ✅ Baseline evaluation
├── llm_based/
│   ├── README.md                       ✅ LLM-based methods doc
│   ├── din_sql/run_din_sql.py          ✅ DIN-SQL runner
│   └── mac_sql/run_mac_sql.py          ✅ MAC-SQL runner
├── sft_based/
│   ├── README.md                       ✅ SFT-based methods doc
│   ├── codes/run_codes.py               ✅ CodeS runner
│   └── qwen_coder/run_qwen_coder.py    ✅ Qwen2.5-Coder runner
└── hybrid/
    ├── README.md                       ✅ Hybrid methods doc
    ├── chess/run_chess.py               ✅ CHESS runner
    └── zero_nl2sql/run_zero_nl2sql.py  ✅ Zero-NL2SQL runner
```

### 3. TACO-SQL Ablation Experiments

```
experiments/taco_sql_exp/
├── README.md                           ✅ TACO-SQL documentation
├── config.py                           ✅ Experiment configuration
├── experiment_runner.py                ✅ Main experiment runner
├── run_ablation.py                     ✅ Unified ablation script
├── __init__.py                         ✅ Package init
├── prompts/                            ✅ Prompt strategies (CRITICAL)
│   ├── __init__.py
│   ├── question_rewriting_prompt.py    ✅ QR prompt strategy
│   ├── query_planning_prompt.py        ✅ QP prompt strategy
│   └── sql_generation_prompt.py        ✅ SQL generation prompts
├── utils/                              ✅ Utility functions
│   ├── __init__.py
│   └── schema_utils.py                 ✅ Schema formatting
├── origin/run_origin.py                ✅ Origin setting
├── qr/run_qr.py                        ✅ QR setting
├── qr_tl/run_qr_tl.py                  ✅ QR+TL setting
└── qr_tl_qp/run_qr_tl_qp.py            ✅ Full TACO-SQL
```

### 4. Evaluation Framework

```
experiments/evaluation/
├── README.md                           ✅ Evaluation documentation
├── evaluation_config.py                ✅ Evaluation configuration
├── metrics_calculator.py               ✅ Metrics calculation
├── error_analysis.py                   ✅ Error analysis tools
├── exec_eval.py                        ✅ Execution evaluation
├── evaluation.py                       ✅ Main evaluation script
├── compare.py                          ✅ Result comparison
├── draw_result.py                      ✅ Visualization
└── tex_table.py                        ✅ LaTeX table generation
```

## ❌ Directories to Exclude (Do NOT Publish)

```
experiments/
├── _internal/                         ❌ Internal development files
│   ├── development_docs/              ❌ Development planning
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

## 📋 File Types to Exclude

- ❌ All `.log` files
- ❌ All `.sh` shell scripts (unless essential)
- ❌ Temporary analysis documents (中文分析文档)
- ❌ Development planning documents
- ❌ Internal checklists and summaries

## 🎯 Critical Files for Reviewers

These files are **essential** and must be included:

### Documentation
1. ✅ `experiments/README.md` - Main documentation
2. ✅ `experiments/TACO-SQL实验核心设置与Prompt策略.md` - Detailed prompt strategies
3. ✅ `experiments/EXPERIMENT_FAIRNESS.md` - Fair comparison principles
4. ✅ `experiments/REVIEWER_GUIDE.md` - Quick reference

### Prompt Strategies (CRITICAL)
1. ✅ `experiments/baselines/base_llm/prompt_strategy.py` - Baseline prompt
2. ✅ `experiments/taco_sql_exp/prompts/question_rewriting_prompt.py` - QR prompt
3. ✅ `experiments/taco_sql_exp/prompts/query_planning_prompt.py` - QP prompt
4. ✅ `experiments/taco_sql_exp/prompts/sql_generation_prompt.py` - SQL prompts

### Core Scripts
1. ✅ `experiments/taco_sql_exp/experiment_runner.py` - Main runner
2. ✅ `experiments/taco_sql_exp/config.py` - Configuration
3. ✅ `experiments/baselines/base_llm/run_experiment.py` - Baseline runner

## 📝 .gitignore Configuration

The `.gitignore` file has been created in `experiments/.gitignore` to automatically exclude:
- `_internal/` directory
- All `.log` files
- All `.sh` scripts
- Temporary analysis documents
- Results directories

## ✅ Pre-Publication Checklist

Before publishing to GitHub:

- [x] Created `_internal/` directory for non-public files
- [x] Moved development docs to `_internal/development_docs/`
- [x] Moved log files to `_internal/logs/`
- [x] Moved temporary analysis to `_internal/temp_analysis/`
- [x] Moved shell scripts to `_internal/scripts/`
- [x] Created `.gitignore` file
- [ ] Verify all README files are in English
- [ ] Verify all code comments are in English
- [ ] Check that all prompt strategies are accessible
- [ ] Ensure experimental settings are clearly documented

## 🚀 Publication Steps

1. **Review the structure**: Check `GITHUB_DIRECTORY_STRUCTURE.md` for complete structure
2. **Verify .gitignore**: Ensure `.gitignore` is in place
3. **Test git status**: Run `git status` to verify only intended files are tracked
4. **Create repository**: Initialize git repository if not already done
5. **Commit and push**: Commit only the public files

## 📊 Summary

### What Reviewers Will See:

✅ **Complete experimental framework** with clear structure
✅ **All prompt strategies** clearly documented and accessible
✅ **Experimental settings** for all 4 TACO-SQL configurations
✅ **Fair comparison principles** explicitly documented
✅ **Evaluation framework** with comprehensive metrics
✅ **All baseline model scripts** for different model types

### What Reviewers Will NOT See:

❌ Development planning documents
❌ Log files and temporary outputs
❌ Internal utility scripts
❌ Temporary analysis documents
❌ Experimental results (unless you choose to include sample)

---

**Status**: ✅ Ready for GitHub publication

