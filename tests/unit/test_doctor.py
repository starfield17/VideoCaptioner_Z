from __future__ import annotations

from pathlib import Path

from captioner.cli.commands.doctor import DoctorOptions, run
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import resolve_app_paths


def test_doctor_reports_all_storage_paths_without_initializing_layout(tmp_path: Path) -> None:
    paths = resolve_app_paths(
        base_dir=tmp_path / "runtime",
        resource_root_override=Path("resources").resolve(),
    )
    service = I18nService("en", resource_dir=paths.i18n_resource_dir)
    payload = run(
        DoctorOptions(locale="en", as_json=True, paths=paths),
        service=service,
    )

    expected = {
        "resource_root",
        "config_dir",
        "data_dir",
        "cache_dir",
        "log_dir",
        "temp_dir",
        "batches_dir",
        "artifacts_dir",
        "models_dir",
        "runtimes_dir",
        "workspaces_dir",
        "downloads_dir",
        "staging_dir",
    }
    assert expected <= payload.keys()
    assert payload["models_dir"] == str(paths.models_dir)
    assert not (paths.data_dir).exists()
