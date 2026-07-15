from __future__ import annotations

from pathlib import Path

from captioner.core.ports.aligner import AlignerPort
from captioner.core.ports.artifact_store import ArtifactStorePort
from captioner.core.ports.asr import ASRPort
from captioner.core.ports.job_store import JobStorePort
from captioner.core.ports.journal import JournalPort
from captioner.core.ports.llm import LLMPort
from captioner.core.ports.media import MediaPort
from captioner.core.ports.runtime import RuntimePort
from captioner.infrastructure.app_paths import resolve_app_paths
from captioner.infrastructure.clock import utc_now
from captioner.infrastructure.ids import new_id
from captioner.infrastructure.logging import configure_logging


def test_port_protocols_and_capability_probe() -> None:
    assert all(protocol is not None for protocol in (ASRPort, AlignerPort, LLMPort, MediaPort))
    assert all(
        protocol is not None
        for protocol in (ArtifactStorePort, JobStorePort, JournalPort, RuntimePort)
    )


def test_infrastructure_helpers(tmp_path: Path) -> None:
    identifier = new_id("job-")
    assert identifier.startswith("job-")
    assert len(identifier) > len("job-")
    assert utc_now().tzinfo is not None
    paths = resolve_app_paths(base_dir=tmp_path)
    logger = configure_logging(paths)
    assert logger.name == "captioner"
    assert paths.log_dir.is_dir()
