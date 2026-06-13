# Contributing to TACO-Benchmark

Thank you for improving TACO-Benchmark. This project welcomes documentation fixes, structural cleanup, new baselines, and bug fixes.

## Before you start

1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for repository layout and change boundaries.
2. Read [docs/README.md](docs/README.md) for the documentation layering convention.
3. Install the environment: `python scripts/setup_env.py` → `taco data download`.

## Documentation conventions

| Layer | Location | Write about |
|:--|:--|:--|
| User guides | `docs/*.md` | Setup, workflows, CLI, concepts |
| Module refs | `**/README.md` next to code | Script args, paths, module-specific notes |
| Reviewer / paper | `experiments/docs/` | Fairness, prompts, reviewer notes |

Avoid duplicating the same quick-start block in both `docs/` and module READMEs — link instead.

## Code conventions

- **Python 3.10+**, English comments and docstrings.
- **Semantic script names** in `benchmark/generation/`; use thin shims when renaming.
- **Paths** — prefer `taco.core.paths` or `Path(__file__)`-based resolution; no hardcoded home directories.
- **Secrets** — never commit real API keys; use `*.example` templates.
- **Artifacts** — do not add generation intermediate files (graphs, large JSON) to git.

## Pull request checklist

- [ ] Changes match the scope of the PR (no unrelated refactors)
- [ ] No secrets or large generated files in the diff
- [ ] User-facing behavior changes are documented in `docs/`
- [ ] Module-only changes update the relevant `README.md`
- [ ] `python -m compileall taco benchmark/generation experiments` passes

## Running checks locally

```bash
source .venv/bin/activate
python -m compileall taco benchmark/generation experiments -q
taco data verify          # after downloading data
bash examples/quick_eval.sh   # optional; requires API key
```

## Questions

Open a GitHub issue for bugs, documentation gaps, or design discussions.

## License

By contributing, you agree that your contributions will be released under the [MIT License](LICENSE).
