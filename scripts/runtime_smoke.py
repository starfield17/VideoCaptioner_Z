"""Install a local Runtime descriptor and run its static/activation Doctor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.cli.commands.runtime import build_manager
from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import ensure_runtime_layout, resolve_app_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--doctor", action="store_true")
    options = parser.parse_args(argv)
    paths = resolve_app_paths()
    ensure_runtime_layout(paths)
    manager = build_manager(paths, activation_client=options.doctor)
    try:
        installation = manager.install(options.descriptor, activate=options.doctor)
        report = manager.doctor(installation.identity, activation=options.doctor)
    except AppError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "runtime": installation.identity.to_dict(),
                "state": installation.state.value,
                "doctor": {
                    "ok": report.ok,
                    "phase": report.phase,
                    "error_code": report.error_code,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
