from __future__ import annotations

import os
from pathlib import Path

from scripts.run_phase2_real_gpu_smoke import cuda_environment


def test_cuda_library_dirs_precede_existing_loader_path(tmp_path: Path) -> None:
    first = tmp_path / "cublas"
    second = tmp_path / "cudnn"
    environment = cuda_environment((first, second))
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep)[:2] == [
        str(first),
        str(second),
    ]


def test_cuda_library_dirs_preserve_empty_loader_path(tmp_path: Path) -> None:
    environment = cuda_environment((tmp_path / "one",))
    assert environment["LD_LIBRARY_PATH"].startswith(str(tmp_path / "one"))
