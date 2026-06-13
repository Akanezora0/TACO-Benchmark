# Examples

Minimal scripts to reproduce common TACO-Benchmark workflows.

## quick_eval.sh

Verify the dataset, run a single base-LLM baseline on the official test split, and print execution-accuracy summary.

```bash
# From repository root (requires API key in configs/llm_config.yaml)
bash examples/quick_eval.sh

# Use a different model
bash examples/quick_eval.sh --model gpt-4o-mini
```

Equivalent manual steps:

```bash
taco data verify
taco eval run --model gpt-4o --dataset beijing
taco eval report --pred experiments/results/baseline_gpt_4o_taco_beijing.json
```

## Further reading

| Doc | Topic |
|:--|:--|
| [docs/INSTALL.md](../docs/INSTALL.md) | Environment setup |
| [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) | Full CLI reference |
| [docs/GENERATION.md](../docs/GENERATION.md) | Data regeneration pipeline |
| [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) | Repository layout |
| [docs/EXAMPLES.md](../docs/EXAMPLES.md) | Challenge NL/SQL examples |
