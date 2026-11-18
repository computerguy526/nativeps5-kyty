# CI

Workflow: `.github/workflows/ci.yml`, on every pull request and push: clang-format on changed lines, the privacy and identity gate (never relaxed), actionlint 1.7.7. The full Kyty build needs GPU/Vulkan/submodules and is not run on hosted runners.

Jobs have `timeout-minutes`, pinned actions (by SHA) and tool versions, and retries (3x, backoff) around network installs.

Rerun: Actions tab, "Re-run failed jobs", or `gh run rerun <run-id> --failed`.

Run the whole workflow locally before every push: `scripts/ci/run-local.sh` (commit first; clean `git worktree` of HEAD, no submodules, fresh venv, same `run:` steps; `--job <id>` for one).
