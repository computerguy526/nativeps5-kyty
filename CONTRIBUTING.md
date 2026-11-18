# Contributing

This repository is a component of [nativeps5](https://github.com/computerguy526/nativeps5). The contribution rules (never any
game data, one change per PR, evidence format, run rules) are in that repository's `CONTRIBUTING.md` and `AGENTS.md`; read them first.
Build and unit tests: [README.md](README.md), "Build and test". The pre-commit hook in `.pre-commit-config.yaml` runs clang-format on `src/`.

## CI (this fork)

Before pushing, run `scripts/ci/run-local.sh` (commit first): it runs every job of `.github/workflows/ci.yml` from a clean checkout of HEAD (same steps, fresh venv, no submodules); `--job <id>` runs one. See docs/ci.md.

Be gentle with GitHub (contributors and agents): no pushes until `scripts/ci/run-local.sh` passes; at most one push per branch, batching commits; do not poll `gh` in loops (look at most every ~5 minutes, or use one `gh run watch <run-id>`); never re-run jobs to test for flakiness (reproduce locally); no bulk API calls. Hammering the API gets the account suspended.

GitHub Actions (`.github/workflows/ci.yml`) runs on every pull request and push to `main`, with no secrets, game data or GPU.
Each check is one command locally (from the repository root):

| Job | What | Locally |
|---|---|---|
| clang-format | the lines the change touches (src/.clang-format); existing code is never reformatted | `pip install clang-format==22.1.3 && git clang-format origin/main -- src` (`--diff` to look) |
| privacy | game data, secrets, personal paths, commit e-mails (a copy of the entry repository's gate, `.github/nps5-ci/`) | `BASE=origin/main TIP=HEAD .github/nps5-ci/privacy.sh` |
| workflows | workflow syntax | `actionlint` |

Not in CI: the build of `kyty_gpu` and the unit tests (14 submodules and a Vulkan stack; a cold build does not fit a basic check; build locally and paste the result), `tests/` formatting (no style file there), and the title runs. A PR needs green CI; it does not replace the evidence in the pull request template.
