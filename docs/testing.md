# Testing and quality gates

`python scripts/check.py --quick` runs formatting, Ruff, Pyright, and unit plus
contract tests. `python scripts/check.py --full` additionally verifies the lock
file, import contracts, i18n catalogs, forbidden patterns, all Phase 0 tests,
and branch coverage with an 85% minimum.

Tests are grouped into `unit`, `property`, `contract`, and `packaging`. Property
tests use Hypothesis for locale and placeholder invariants. Packaging tests
inspect commands and layouts without compiling Nuitka; the local build wrapper
then performs the real platform smoke build.

Before submitting a patch, run:

```bash
uv sync --frozen
uv run python scripts/check.py --full
uv run pytest tests/packaging -q
uv run python main.py --cli --help
uv run python main.py --cli doctor --json
QT_QPA_PLATFORM=offscreen uv run python main.py --gui --smoke-test
uv run python scripts/build_nuitka.py --clean --version 0.0.0
```
