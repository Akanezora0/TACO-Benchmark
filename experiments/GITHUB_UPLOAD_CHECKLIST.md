# GitHub Upload Checklist

## ✅ Directories to Upload to GitHub

### 1. Root Level Documentation
```
experiments/
├── README.md                           ✅ Main documentation
├── EXPERIMENT_FAIRNESS.md              ✅ Fair comparison principles
├── FINAL_VERSION_SUMMARY.md            ✅ Complete summary
├── REVIEWER_GUIDE.md                   ✅ Quick reviewer guide
├── TACO-SQL实验核心设置与Prompt策略.md    ✅ Detailed prompt strategies
└── .gitignore                          ✅ Git ignore rules
```

### 2. Baseline Experiments (`baselines/`)
```
baselines/
├── README.md                           ✅
├── run_all_baselines.py                ✅
├── base_llm/
│   ├── README.md                       ✅
│   ├── prompt_strategy.py              ✅ CRITICAL: Baseline prompt
│   ├── model_wrapper.py                ✅
│   ├── run_experiment.py               ✅
│   ├── batch_evaluate.py               ✅
│   ├── experiment_config.py            ✅
│   └── evaluate_baseline.py            ✅
├── llm_based/
│   ├── README.md                       ✅
│   ├── din_sql/run_din_sql.py          ✅
│   └── mac_sql/run_mac_sql.py          ✅
├── sft_based/
│   ├── README.md                       ✅
│   ├── codes/run_codes.py               ✅
│   └── qwen_coder/run_qwen_coder.py    ✅
└── hybrid/
    ├── README.md                       ✅
    ├── chess/run_chess.py               ✅
    └── zero_nl2sql/run_zero_nl2sql.py  ✅
```

### 3. TACO-SQL Experiments (`taco_sql_exp/`)
```
taco_sql_exp/
├── README.md                           ✅
├── config.py                           ✅
├── experiment_runner.py                ✅
├── run_ablation.py                     ✅
├── __init__.py                         ✅
├── prompts/                            ✅ CRITICAL: All prompt strategies
│   ├── __init__.py                     ✅
│   ├── question_rewriting_prompt.py    ✅ QR prompt
│   ├── query_planning_prompt.py        ✅ QP prompt
│   └── sql_generation_prompt.py        ✅ SQL prompts
├── utils/
│   ├── __init__.py                     ✅
│   └── schema_utils.py                 ✅
├── origin/run_origin.py                ✅
├── qr/run_qr.py                        ✅
├── qr_tl/run_qr_tl.py                  ✅
└── qr_tl_qp/run_qr_tl_qp.py            ✅
```

### 4. Evaluation Framework (`evaluation/`)
```
evaluation/
├── README.md                           ✅
├── evaluation_config.py                ✅
├── metrics_calculator.py               ✅
├── error_analysis.py                   ✅
├── exec_eval.py                        ✅
├── evaluation.py                       ✅
├── compare.py                          ✅
├── draw_result.py                      ✅
└── tex_table.py                        ✅
```

## ❌ Do NOT Upload

### Excluded Directories
```
experiments/
├── _internal/                         ❌ Internal files (already moved)
│   ├── development_docs/              ❌
│   ├── logs/                          ❌
│   ├── temp_analysis/                 ❌
│   └── scripts/                       ❌
│
├── results/                           ❌ Optional (can include sample)
│   ├── raw/                           ❌
│   ├── processed/                     ❌
│   └── visualizations/                ❌
│
└── scripts/                           ❌ Internal scripts
```

### Excluded File Types
- ❌ All `.log` files
- ❌ All `.sh` shell scripts
- ❌ Temporary analysis documents (中文)
- ❌ Development planning documents

## 🎯 Critical Files for Reviewers

### Must-Have Documentation
1. ✅ `experiments/README.md` - Main entry point
2. ✅ `experiments/TACO-SQL实验核心设置与Prompt策略.md` - Detailed prompts
3. ✅ `experiments/EXPERIMENT_FAIRNESS.md` - Fair comparison
4. ✅ `experiments/REVIEWER_GUIDE.md` - Quick reference

### Must-Have Prompt Strategies
1. ✅ `baselines/base_llm/prompt_strategy.py` - Baseline prompt
2. ✅ `taco_sql_exp/prompts/question_rewriting_prompt.py` - QR prompt
3. ✅ `taco_sql_exp/prompts/query_planning_prompt.py` - QP prompt
4. ✅ `taco_sql_exp/prompts/sql_generation_prompt.py` - SQL prompts

### Must-Have Core Scripts
1. ✅ `taco_sql_exp/experiment_runner.py` - Main runner
2. ✅ `taco_sql_exp/config.py` - Configuration
3. ✅ `baselines/base_llm/run_experiment.py` - Baseline runner

## 📋 Quick Upload Command

```bash
# Navigate to experiments directory
cd experiments

# Verify .gitignore is working
git status

# Should see only:
# - README.md
# - EXPERIMENT_FAIRNESS.md
# - FINAL_VERSION_SUMMARY.md
# - REVIEWER_GUIDE.md
# - TACO-SQL实验核心设置与Prompt策略.md
# - baselines/
# - taco_sql_exp/
# - evaluation/
# - .gitignore

# Should NOT see:
# - _internal/
# - *.log files
# - *.sh files
# - results/ (unless you want to include)
```

## ✅ Final Verification

Before uploading, verify:

- [x] `_internal/` directory exists and contains non-public files
- [x] `.gitignore` file is in place
- [x] All README files are in English
- [x] All prompt strategies are accessible
- [x] Experimental settings are clearly documented
- [x] Fair comparison principles are documented
- [ ] Run `git status` to verify only intended files are tracked

## 📊 Summary

**Total Directories to Upload**: 3 main directories
- `baselines/` - Baseline experiments
- `taco_sql_exp/` - TACO-SQL ablation experiments
- `evaluation/` - Evaluation framework

**Total Documentation Files**: 5 core documents
- Main README
- Fair comparison principles
- Detailed prompt strategies
- Reviewer guide
- Framework summary

**Status**: ✅ Ready for GitHub upload

