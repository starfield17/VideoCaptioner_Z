from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest

from captioner.adapters.llm.fake import (
    ScriptedCancellation,
    ScriptedCrash,
    ScriptedCrashError,
    ScriptedJSON,
)
from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    LLMItem,
    LLMRequest,
    SourceCorrectionResponse,
    response_batch_schema,
)


def _request() -> LLMRequest:
    return LLMRequest("correct_source", (LLMItem("item-1", "source"),))


def test_scripted_adapter_returns_structured_success_and_malformed_json() -> None:
    adapter = ScriptedLLMAdapter(
        structured_responses=[
            SourceCorrectionResponse("item-1", "corrected"),
            ScriptedJSON("not json"),
        ]
    )
    first = asyncio.run(
        adapter.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
    )
    assert first == SourceCorrectionResponse("item-1", "corrected")
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        asyncio.run(
            adapter.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    assert len(adapter.structured_calls) == 2


def test_scripted_batch_can_express_id_variants() -> None:
    schema = response_batch_schema(SourceCorrectionResponse)
    adapter = ScriptedLLMAdapter(
        structured_responses=[
            ScriptedJSON(
                json.dumps(
                    [
                        {"id": "extra", "corrected_source": "corrected"},
                        {"id": "item-1", "corrected_source": "corrected"},
                    ]
                )
            )
        ]
    )
    result = asyncio.run(adapter.generate_structured(_request(), schema, ExecutionContext()))
    responses = cast(tuple[SourceCorrectionResponse, ...], result.responses)
    assert [item.id for item in responses] == ["extra", "item-1"]


def test_scripted_cancellation_and_injected_crash_are_distinct() -> None:
    cancelled = ScriptedLLMAdapter(structured_responses=[ScriptedCancellation("checkpoint")])
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(
            cancelled.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    crashed = ScriptedLLMAdapter(structured_responses=[ScriptedCrash("checkpoint")])
    with pytest.raises(ScriptedCrashError) as raised:
        asyncio.run(
            crashed.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    assert raised.value.checkpoint == "checkpoint"
