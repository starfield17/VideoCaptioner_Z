"""Production and test-double token counters for LLM request budgeting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, NoReturn, cast

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import LLM_RESPONSE_SCHEMA_VERSION, LLMRequest, encode_llm_request
from captioner.core.ports.token_counter import TokenCounter

SUPPORTED_TOKENIZERS: Final[frozenset[str]] = frozenset({"cl100k_base", "o200k_base"})
TOKENIZER_MANIFEST_FILENAME: Final[str] = "tokenizer-manifest.json"

_ENCODING_SOURCE = "https://openaipublic.blob.core.windows.net/encodings"
_TIKTOKEN_UPSTREAM_VERSION = "0.13.0"
_ENCODING_PARAMS: Final[Mapping[str, Mapping[str, object]]] = {
    "cl100k_base": {
        "name": "cl100k_base",
        "pat_str": (
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|"
            r"\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|"
            r"\s*[\r\n]|\s+(?!\S)|\s"
        ),
        "special_tokens": {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    },
    "o200k_base": {
        "name": "o200k_base",
        "pat_str": "|".join(
            (
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"\p{N}{1,3}",
                r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
                r"\s*[\r\n]+",
                r"\s+(?!\S)",
                r"\s+",
            )
        ),
        "special_tokens": {"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
    },
}


def resolve_tokenizer_id(tokenizer: str, model: str) -> str:
    """Resolve a configured tokenizer using the pinned tiktoken model map."""
    if not tokenizer.strip():
        raise AppError("llm.tokenizer_unknown", {"tokenizer": tokenizer})
    selected = tokenizer.strip()
    if selected == "auto":
        if not model.strip():
            raise AppError("llm.tokenizer_unknown", {"model": model})
        try:
            resolved = tiktoken.encoding_name_for_model(model.strip())
        except (KeyError, ValueError) as exc:
            raise AppError("llm.tokenizer_unknown", {"model": model.strip()}) from exc
        if resolved not in SUPPORTED_TOKENIZERS:
            raise AppError("llm.tokenizer_unknown", {"tokenizer": resolved})
        return resolved
    if selected not in SUPPORTED_TOKENIZERS:
        raise AppError("llm.tokenizer_unknown", {"tokenizer": selected})
    return selected


class ModelTokenCounter:
    """Production counter backed by verified, bundled tiktoken data."""

    def __init__(self, tokenizer_id: str, *, resource_dir: Path | None = None) -> None:
        if tokenizer_id not in SUPPORTED_TOKENIZERS:
            raise AppError("llm.tokenizer_unknown", {"tokenizer": tokenizer_id})
        selected_dir = resource_dir
        if selected_dir is None:
            from captioner.infrastructure.app_paths import resolve_app_paths

            selected_dir = resolve_app_paths().tokenizer_resource_dir
        self._tokenizer_id = tokenizer_id
        self._encoding = _load_packaged_encoding(tokenizer_id, selected_dir)

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


class SerializedRequestTokenCounter:
    """Estimate the full wire request with the same serialization as the adapter."""

    def __init__(
        self,
        token_counter: TokenCounter,
        model: str,
        temperature: float,
        *,
        response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION,
    ) -> None:
        self._token_counter = token_counter
        self._model = model
        self._temperature = temperature
        self._response_schema_version = response_schema_version

    def estimate_input_tokens(self, request: LLMRequest, response_schema: type[object]) -> int:
        encoded = encode_llm_request(
            request,
            self._model,
            self._temperature,
            response_schema,
            response_schema_version=self._response_schema_version,
        )
        return self._token_counter.count(encoded.decode("utf-8"))


def _load_packaged_encoding(tokenizer_id: str, resource_dir: Path) -> tiktoken.Encoding:
    manifest_path = resource_dir / TOKENIZER_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        encodings = manifest["encodings"]
        entry = next(item for item in encodings if item["encoding_id"] == tokenizer_id)
        filename = entry["resource_filename"]
        expected_sha256 = entry["expected_sha256"]
        if (
            set(entry)
            != {
                "encoding_id",
                "resource_filename",
                "expected_sha256",
                "upstream_source",
                "upstream_version",
            }
            or entry["upstream_source"] != f"{_ENCODING_SOURCE}/{filename}"
            or entry["upstream_version"] != _TIKTOKEN_UPSTREAM_VERSION
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected_sha256, str)
        ):
            _raise_invalid_resource_manifest()
        resource_path = (resource_dir / filename).resolve()
        if resource_path.parent != resource_dir.resolve():
            _raise_invalid_resource_manifest()
        digest = hashlib.sha256(resource_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            _raise_invalid_resource_manifest()
        bpe = load_tiktoken_bpe(str(resource_path), expected_hash=expected_sha256)
        params = _ENCODING_PARAMS[tokenizer_id]
        return tiktoken.Encoding(
            name=cast(str, params["name"]),
            pat_str=cast(str, params["pat_str"]),
            mergeable_ranks=bpe,
            special_tokens=cast(dict[str, int], params["special_tokens"]),
        )
    except (KeyError, OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AppError("llm.tokenizer_unknown", {"tokenizer": tokenizer_id}) from exc


def _raise_invalid_resource_manifest() -> NoReturn:
    raise ValueError
