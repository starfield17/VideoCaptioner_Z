"""Unit tests for RecoveryDialog."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from captioner.core.application.recovery import RecoveryItem
from captioner.core.domain.batch import BatchState
from captioner.gui.widgets.recovery_dialog import RecoveryDialog
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-recovery-dialog"])


def test_recovery_dialog_widgets() -> None:
    service = I18nService("en")
    items = (
        RecoveryItem(
            batch_id="batch-a",
            created_at_utc="t0",
            state=BatchState.PENDING,
            job_count=1,
            pause_requested=False,
            missing_input_paths=(),
            last_event_seq=1,
            blocked_code=None,
        ),
    )
    dialog = RecoveryDialog(service, items)
    for name in (
        "recoveryDialog",
        "recoveryTable",
        "recoveryResumeButton",
        "recoveryCancelButton",
        "recoveryLaterButton",
        "recoveryFailureLabel",
    ):
        if name == "recoveryDialog":
            assert dialog.objectName() == name
        else:
            assert dialog.findChild(QObject, name) is not None, name


def test_recovery_dialog_actions() -> None:
    service = I18nService("en")
    items = (
        RecoveryItem(
            batch_id="batch-a",
            created_at_utc="t0",
            state=BatchState.PENDING,
            job_count=1,
            pause_requested=True,
            missing_input_paths=("/missing/a.wav", "/missing/b.wav"),
            last_event_seq=1,
            blocked_code="recovery.input_missing",
        ),
        RecoveryItem(
            batch_id="batch-b",
            created_at_utc="t1",
            state=BatchState.INTERRUPTED,
            job_count=2,
            pause_requested=False,
            missing_input_paths=(),
            last_event_seq=2,
            blocked_code=None,
        ),
    )
    dialog = RecoveryDialog(service, items)
    dialog._table.selectRow(0)  # type: ignore[attr-defined]
    dialog._on_selection()  # type: ignore[attr-defined]
    assert dialog.selected_batch_id == "batch-a"
    # blocked resume
    dialog._on_resume()  # type: ignore[attr-defined]
    assert dialog.action is None
    dialog._on_cancel()  # type: ignore[attr-defined]
    assert dialog.action == "cancel"
    dialog2 = RecoveryDialog(service, items)
    dialog2._table.selectRow(1)  # type: ignore[attr-defined]
    dialog2._on_selection()  # type: ignore[attr-defined]
    dialog2._on_resume()  # type: ignore[attr-defined]
    assert dialog2.action == "resume"
    dialog3 = RecoveryDialog(service, items)
    dialog3._on_later()  # type: ignore[attr-defined]
    assert dialog3.action == "later"
