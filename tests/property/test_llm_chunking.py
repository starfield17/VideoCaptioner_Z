from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner


@dataclass(frozen=True, slots=True)
class FakeCounter:
    def count(self, text: str) -> int:
        return len(text.split())


@given(st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=30))
def test_chunk_planner_is_forward_only_and_repeatable(widths: list[int]) -> None:
    items = tuple(
        ChunkItem(f"unit-{index}", " ".join("x" for _ in range(width)))
        for index, width in enumerate(widths)
    )
    config = ChunkingConfig(
        max_items=4,
        max_input_tokens=12,
        context_before_items=2,
        context_after_items=2,
    )
    planner = ChunkPlanner(FakeCounter(), config)
    first = planner.plan(items)
    second = planner.plan(items)
    assert first == second
    assert tuple(item_id for chunk in first for item_id in chunk.item_ids) == tuple(
        item.id for item in items
    )
    for chunk in first:
        assert not set(chunk.item_ids) & set(chunk.context_ids)
        assert len(chunk.items) <= config.max_items
        assert (
            sum(FakeCounter().count(item.text) for item in chunk.items) <= config.max_input_tokens
        )


def test_context_budget_and_audio_budget_are_both_enforced() -> None:
    items = tuple(
        ChunkItem(f"unit-{index}", "word", index * 1000, index * 1000 + 100) for index in range(5)
    )
    planner = ChunkPlanner(
        FakeCounter(),
        ChunkingConfig(
            max_items=2,
            max_input_tokens=4,
            context_before_items=2,
            context_after_items=2,
            max_audio_context_duration_ms=2200,
        ),
    )
    chunks = planner.plan(items)
    assert tuple(item.id for chunk in chunks for item in chunk.items) == tuple(
        item.id for item in items
    )
    for chunk in chunks:
        window = (*chunk.context, *chunk.items)
        assert max(item.end_ms for item in window) - min(item.start_ms for item in window) <= 2200


def test_single_item_over_budget_has_structured_error() -> None:
    from pytest import raises

    from captioner.core.domain.errors import AppError
    from captioner.core.policies.llm_chunking import plan_chunks

    with raises(AppError, match=r"llm\.item_too_large"):
        plan_chunks(
            (ChunkItem("too-large", "one two three"),),
            ChunkingConfig(max_input_tokens=2),
            FakeCounter(),
        )
