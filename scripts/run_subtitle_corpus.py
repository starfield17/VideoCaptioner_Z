"""Run the deterministic subtitle processing corpus without network access."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from captioner.adapters.subtitles.corpus import run_project_subtitle_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_directory", type=Path)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(None if argv is None else list(argv))
    report = run_project_subtitle_corpus(arguments.fixture_directory)
    if arguments.json:
        print(
            json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    else:
        for fixture in report.fixtures:
            print(
                f"{fixture.name}: cues={fixture.cue_count} "
                f"max_cps_milli={fixture.max_cps_milli} "
                f"max_line_width={fixture.max_line_width} "
                f"warnings={len(fixture.warnings)} errors={len(fixture.errors)}"
            )
        for error in report.errors:
            print(f"corpus failed: {error}", file=sys.stderr)
    return int(report.failed != 0 or bool(report.errors))


if __name__ == "__main__":
    raise SystemExit(main())
