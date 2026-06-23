
# CHANGELOG — Layer 2 (LLM Providers)

## Fixed

### Critical: JSON array extraction regex
- **File:** `llm/provider.py`
- **Bug:** `_JSON_ARRAY_RE` used `$$$$.*$$` which matches literal `` characters,
  not `[` and `]`. JSON arrays embedded in LLM output outside of markdown
  fences could never be extracted (strategy #3 was broken for arrays).
- **Fix:** Changed regex to `\[.*\]`.
- **Impact:** All array-shaped LLM responses that were previously falling through
  to `extract_json() -> None` now parse correctly.

### System prompt accumulation on retry
- **File:** `llm/provider.py`
- **Bug:** `generate_json()` mutated copies derived from the request after
  previous retries had already appended the retry suffix. While `model_copy()`
  prevented mutation of the original, each retry copy was derived from the
  (already-suffixed) request rather than the original prompt. After N retries,
  the system prompt contained N copies of the retry instruction.
- **Fix:** Store `original_system = request.system_prompt` before the retry loop.
  Retry copies append the suffix to the original, not to the accumulated chain.
- **Impact:** Cleaner prompts sent to LLM on retries 2+; slight reduction in
  token waste on retries.

## Added

### Test coverage expansion (`tests/unit/test_llm.py`)
- `test_json_array_embedded_in_text` — validates array regex fix
- `test_json_array_in_fence` — arrays inside markdown code blocks
- `test_prefers_direct_over_fence` — extraction priority ordering
- `test_handles_whitespace_around_fence` — padded fence content
- `test_retry_failure_with_provider_error` — LLMError propagates from generate()
- `test_retry_failure_with_rate_limit` — LLMRateLimitError propagates correctly
- `test_system_prompt_not_mutated` — original request is never modified
- `test_array_response_wrapped_in_items` — generate_json wraps arrays per contract
- `test_retries_zero_still_runs_once` — edge case: retries=0 clamps to 1

### New mock: `_FailingMockProvider`
- Raises a configurable exception on every `generate()` call
- Used to test exception propagation through `generate_json()`

## Not Changed (by design)

- Provider implementations (`claude_provider.py`, `openai_provider.py`,
  `gemini_provider.py`, `ollama_provider.py`) — no bugs found in concrete
  providers. Exception mapping (`APIError -> LLMError`, `RateLimitError ->
  LLMRateLimitError`) is correct in all four.
- Factory (`factory.py`) — match/case dispatch is correct.
- Prompt builders (`prompts/`) — no bugs found.
- Public interfaces — all preserved unchanged.
# CHANGELOG — Layer 2 (LLM Providers)

## Fixed

### Critical: JSON array extraction regex
- **File:** `llm/provider.py`
- **Bug:** `_JSON_ARRAY_RE` used `$$$$.*$$` which matches literal `` characters,
  not `[` and `]`. JSON arrays embedded in LLM output outside of markdown
  fences could never be extracted (strategy #3 was broken for arrays).
- **Fix:** Changed regex to `\[.*\]`.
- **Impact:** All array-shaped LLM responses that were previously falling through
  to `extract_json() -> None` now parse correctly.

### System prompt accumulation on retry
- **File:** `llm/provider.py`
- **Bug:** `generate_json()` mutated copies derived from the request after
  previous retries had already appended the retry suffix. While `model_copy()`
  prevented mutation of the original, each retry copy was derived from the
  (already-suffixed) request rather than the original prompt. After N retries,
  the system prompt contained N copies of the retry instruction.
- **Fix:** Store `original_system = request.system_prompt` before the retry loop.
  Retry copies append the suffix to the original, not to the accumulated chain.
- **Impact:** Cleaner prompts sent to LLM on retries 2+; slight reduction in
  token waste on retries.

## Added

### Test coverage expansion (`tests/unit/test_llm.py`)
- `test_json_array_embedded_in_text` — validates array regex fix
- `test_json_array_in_fence` — arrays inside markdown code blocks
- `test_prefers_direct_over_fence` — extraction priority ordering
- `test_handles_whitespace_around_fence` — padded fence content
- `test_retry_failure_with_provider_error` — LLMError propagates from generate()
- `test_retry_failure_with_rate_limit` — LLMRateLimitError propagates correctly
- `test_system_prompt_not_mutated` — original request is never modified
- `test_array_response_wrapped_in_items` — generate_json wraps arrays per contract
- `test_retries_zero_still_runs_once` — edge case: retries=0 clamps to 1

### New mock: `_FailingMockProvider`
- Raises a configurable exception on every `generate()` call
- Used to test exception propagation through `generate_json()`

## Not Changed (by design)

- Provider implementations (`claude_provider.py`, `openai_provider.py`,
  `gemini_provider.py`, `ollama_provider.py`) — no bugs found in concrete
  providers. Exception mapping (`APIError -> LLMError`, `RateLimitError ->
  LLMRateLimitError`) is correct in all four.
- Factory (`factory.py`) — match/case dispatch is correct.
- Prompt builders (`prompts/`) — no bugs found.
- Public interfaces — all preserved unchanged.


Summary of Changes

File	Change	Severity
llm/provider.py	Fixed _JSON_ARRAY_RE: $$$$.*$$ → $$.*$$	Critical
llm/provider.py	Preserve original system prompt across retries	Minor
tests/unit/test_llm.py	+9 new test cases, +1 new mock class	Coverage
CHANGELOG_LAYER2.md	New file documenting all changes	Documentation

No architecture changes, no new abstractions, no public interface modifications. The provider exception hierarchy (LLMError ← LLMRateLimitError) was already correct across all four providers and required no fixes.


Fix 1: Greedy Regex → Balanced Brace Scanner

The $$.*$$ fix from earlier was only half the problem. The remaining {.*} and [.*] regexes are greedy and will over-match when multiple JSON values appear in text. Worse, even non-greedy .*? fails on nested structures like {"a": {"b": 1}} — it would match at the inner closing brace, producing invalid JSON.


The only correct fix is a character-level scanner that counts nesting depth and respects string literals.


Fix 2: Unused LLMResponseParseError

The exception is imported but never raised. On exhaustion, the code silently returns a response with json_valid = False, leaving callers to check a boolean flag. This makes the contract ambiguous. The fix is to raise the exception, making the contract explicit: generate_json() returns a successfully parsed response or raises.


Fix 3: Array Wrapping

Confirmed: the TRANSFORM_RESPONSE_SCHEMA in schemas.py requires {"analysis_summary": ..., "transforms": [...]} — always a dict. The {"items": array} wrapping in generate_json() is a defensive fallback that should never trigger in normal operation. I'll add a comment documenting this contract.



Modified Files

cuda_opt_agent/llm/provider.py

python
"""Abstract LLM provider with concrete JSON-generation retry logic.

Each provider implements generate(). The base class handles generate_json()
with extraction + retry so that logic is never duplicated.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from cuda_opt_agent.core.exceptions import LLMError, LLMResponseParseError
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse


# ── JSON extraction helpers ────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_balanced(text: str, open_char: str, close_char: str) -> str | None:
    """Extract the first balanced substring delimited by *open_char* / *close_char*.

    Counts nesting depth and respects JSON string literals (including backslash
    escapes).  Returns the substring from the first *open_char* through its
    matching *close_char*, or ``None`` if no balanced substring is found.

    This replaces the previous greedy-regex approach (``r"\\{.*\\}"``) which
    over-matched when multiple JSON values appeared in the same text or when
    values contained nested objects/arrays.
    """
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if in_string:
            if c == "\\":
                escape_next = True
            elif c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
        elif c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort extraction of a JSON value from LLM output.

    Tries, in order:
    1. Direct parse of entire text.
    2. Parse content inside ```json ... ``` or ``` ... ``` fences.
    3. First balanced ``{ ... }`` or ``[ ... ]`` substring (respects nesting
       and string literals — no greedy over-match).
    """
    text = text.strip()
    if not text:
        return None

    # 1. Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced JSON value
    for open_c, close_c in (("{", "}"), ("[", "]")):
        candidate = _extract_balanced(text, open_c, close_c)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


# ── Abstract Base ──────────────────────────────────────────────

class LLMProvider(ABC):
    """Base class for all LLM providers.

    Subclasses implement: generate(), health_check(), estimate_cost(), provider_name().
    The base class provides generate_json() with automatic retry on parse failure.
    """

    # ── Abstract interface ─────────────────────────────────────

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a text completion. Raises LLMError on API failure."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and credentials are valid."""

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for the given token counts."""

    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g., 'claude', 'openai')."""

    # ── Concrete: JSON generation with retry ───────────────────

    async def generate_json(
        self, request: LLMRequest, retries: int = 3
    ) -> LLMResponse:
        """Generate and parse JSON output. Retries on parse failure.

        Contract:
            Returns an ``LLMResponse`` whose ``parsed_json`` is a ``dict``.
            If the LLM returns a bare JSON array it is wrapped as
            ``{"items": [...]}`` so that callers can always expect a dict
            (matching ``TRANSFORM_RESPONSE_SCHEMA``).

            Raises ``LLMResponseParseError`` if parsing fails after all
            retries.  Raises ``LLMError`` (or a subclass such as
            ``LLMRateLimitError``) if the underlying provider call fails.
        """
        last_response: LLMResponse | None = None
        original_system = request.system_prompt

        for attempt in range(max(1, retries)):
            req = request.model_copy()

            if attempt > 0:
                suffix = (
                    "\n\nIMPORTANT: Your previous response contained invalid JSON. "
                    "Respond with ONLY a valid JSON object. No markdown fences, "
                    "no explanation, no text outside the JSON."
                )
                req.system_prompt = (original_system or "") + suffix

            req.json_mode = True
            response = await self.generate(req)

            parsed = extract_json(response.content)
            if parsed is not None and isinstance(parsed, (dict, list)):
                response.json_valid = True
                # Bare arrays are wrapped so callers always receive a dict.
                # In normal operation the LLM returns a dict matching
                # TRANSFORM_RESPONSE_SCHEMA; this branch is a defensive
                # fallback for edge-case LLM outputs.
                response.parsed_json = (
                    parsed if isinstance(parsed, dict) else {"items": parsed}
                )
                return response

            response.json_valid = False
            last_response = response

        if last_response is not None:
            raise LLMResponseParseError(
                f"Failed to parse JSON after {max(1, retries)} attempt(s). "
                f"Last response (truncated): {last_response.content[:200]!r}"
            )

        raise LLMError("generate_json produced no response")
"""Abstract LLM provider with concrete JSON-generation retry logic.

Each provider implements generate(). The base class handles generate_json()
with extraction + retry so that logic is never duplicated.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from cuda_opt_agent.core.exceptions import LLMError, LLMResponseParseError
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse


# ── JSON extraction helpers ────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_balanced(text: str, open_char: str, close_char: str) -> str | None:
    """Extract the first balanced substring delimited by *open_char* / *close_char*.

    Counts nesting depth and respects JSON string literals (including backslash
    escapes).  Returns the substring from the first *open_char* through its
    matching *close_char*, or ``None`` if no balanced substring is found.

    This replaces the previous greedy-regex approach (``r"\\{.*\\}"``) which
    over-matched when multiple JSON values appeared in the same text or when
    values contained nested objects/arrays.
    """
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if in_string:
            if c == "\\":
                escape_next = True
            elif c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
        elif c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort extraction of a JSON value from LLM output.

    Tries, in order:
    1. Direct parse of entire text.
    2. Parse content inside ```json ... ``` or ``` ... ``` fences.
    3. First balanced ``{ ... }`` or ``[ ... ]`` substring (respects nesting
       and string literals — no greedy over-match).
    """
    text = text.strip()
    if not text:
        return None

    # 1. Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced JSON value
    for open_c, close_c in (("{", "}"), ("[", "]")):
        candidate = _extract_balanced(text, open_c, close_c)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


# ── Abstract Base ──────────────────────────────────────────────

class LLMProvider(ABC):
    """Base class for all LLM providers.

    Subclasses implement: generate(), health_check(), estimate_cost(), provider_name().
    The base class provides generate_json() with automatic retry on parse failure.
    """

    # ── Abstract interface ─────────────────────────────────────

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a text completion. Raises LLMError on API failure."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and credentials are valid."""

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for the given token counts."""

    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g., 'claude', 'openai')."""

    # ── Concrete: JSON generation with retry ───────────────────

    async def generate_json(
        self, request: LLMRequest, retries: int = 3
    ) -> LLMResponse:
        """Generate and parse JSON output. Retries on parse failure.

        Contract:
            Returns an ``LLMResponse`` whose ``parsed_json`` is a ``dict``.
            If the LLM returns a bare JSON array it is wrapped as
            ``{"items": [...]}`` so that callers can always expect a dict
            (matching ``TRANSFORM_RESPONSE_SCHEMA``).

            Raises ``LLMResponseParseError`` if parsing fails after all
            retries.  Raises ``LLMError`` (or a subclass such as
            ``LLMRateLimitError``) if the underlying provider call fails.
        """
        last_response: LLMResponse | None = None
        original_system = request.system_prompt

        for attempt in range(max(1, retries)):
            req = request.model_copy()

            if attempt > 0:
                suffix = (
                    "\n\nIMPORTANT: Your previous response contained invalid JSON. "
                    "Respond with ONLY a valid JSON object. No markdown fences, "
                    "no explanation, no text outside the JSON."
                )
                req.system_prompt = (original_system or "") + suffix

            req.json_mode = True
            response = await self.generate(req)

            parsed = extract_json(response.content)
            if parsed is not None and isinstance(parsed, (dict, list)):
                response.json_valid = True
                # Bare arrays are wrapped so callers always receive a dict.
                # In normal operation the LLM returns a dict matching
                # TRANSFORM_RESPONSE_SCHEMA; this branch is a defensive
                # fallback for edge-case LLM outputs.
                response.parsed_json = (
                    parsed if isinstance(parsed, dict) else {"items": parsed}
                )
                return response

            response.json_valid = False
            last_response = response

        if last_response is not None:
            raise LLMResponseParseError(
                f"Failed to parse JSON after {max(1, retries)} attempt(s). "
                f"Last response (truncated): {last_response.content[:200]!r}"
            )

        raise LLMError("generate_json produced no response")

Changes from the previous version:


What	Before	After
Strategy #3 extraction	Greedy regex r"\{.*\}" / r"$$.*$$"	_extract_balanced() — character-level scanner with depth counting and string-literal awareness
Unused regex constants	_JSON_OBJECT_RE, _JSON_ARRAY_RE defined	Removed (replaced by _extract_balanced)
LLMResponseParseError	Imported, never raised	Raised on retry exhaustion with truncated last-response content
Original system prompt	Already fixed in prior pass	Unchanged (still correct)
Array wrapping	Unchanged	Added contract docstring explaining {"items": ...} fallback and its relationship to TRANSFORM_RESPONSE_SCHEMA


tests/unit/test_llm.py

python
"""Tests for LLM layer: provider ABC, JSON extraction, prompts, factory."""

from __future__ import annotations

import json

import pytest

from cuda_opt_agent.core.exceptions import (
    ConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
)
from cuda_opt_agent.llm.provider import LLMProvider, extract_json, _extract_balanced
from cuda_opt_agent.llm.factory import create_provider
from cuda_opt_agent.core.config import LLMConfig, LLMProviderConfig
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics
from cuda_opt_agent.llm.prompts.transform import build_transform_prompt
from cuda_opt_agent.llm.prompts.base import (
    format_bottleneck,
    format_kernel_source,
    format_metrics,
)


# ── JSON Extraction Tests ──────────────────────────────────────

class TestExtractJson:

    def test_direct_json_object(self):
        text = '{"transforms": [], "analysis_summary": "ok"}'
        result = extract_json(text)
        assert result is not None
        assert "transforms" in result

    def test_direct_json_array(self):
        text = '[1, 2, 3]'
        result = extract_json(text)
        assert result is not None
        assert result == [1, 2, 3]

    def test_json_in_markdown_fence(self):
        text = 'Here is the analysis:\n```json\n{"ok": true}\n```\nDone.'
        result = extract_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_json_in_plain_fence(self):
        text = '```{"ok": true}\n```'
        result = extract_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_json_embedded_in_text(self):
        text = 'Based on analysis, the result is {"speedup": 2.5} which is good.'
        result = extract_json(text)
        assert result is not None
        assert result["speedup"] == 2.5

    def test_json_array_embedded_in_text(self):
        text = 'The results are [10, 20, 30] as expected.'
        result = extract_json(text)
        assert result is not None
        assert result == [10, 20, 30]

    def test_json_array_in_fence(self):
        text = 'Here are the items:\n```json\n["a", "b", "c"]\n```'
        result = extract_json(text)
        assert result is not None
        assert result == ["a", "b", "c"]

    def test_empty_string(self):
        assert extract_json("") is None
        assert extract_json("  ") is None

    def test_no_json(self):
        assert extract_json("This is just plain text with no JSON.") is None

    def test_invalid_json(self):
        assert extract_json("{not valid json}") is None

    def test_nested_json(self):
        obj = {
            "transforms": [
                {"name": "tiling", "params": {"tile_size": 16}},
                {"name": "unrolling", "params": {"factor": 4}},
            ],
            "warnings": ["may increase register pressure"],
        }
        text = f"Analysis complete:\n```json\n{json.dumps(obj)}\n```"
        result = extract_json(text)
        assert result is not None
        assert len(result["transforms"]) == 2
        assert result["transforms"][0]["params"]["tile_size"] == 16

    def test_prefers_direct_over_fence(self):
        text = '{"direct": true}'
        result = extract_json(text)
        assert result is not None
        assert result["direct"] is True

    def test_handles_whitespace_around_fence(self):
        text = '  \n  ```json\n{"padded": true}\n```  \n  '
        result = extract_json(text)
        assert result is not None
        assert result["padded"] is True

    def test_multiple_objects_picks_first(self):
        """Greedy regex would over-match. Balanced scanner picks the first complete value."""
        text = 'prefix {"a": 1} middle {"b": 2} suffix'
        result = extract_json(text)
        assert result is not None
        assert result == {"a": 1}

    def test_multiple_arrays_picks_first(self):
        text = 'before [1, 2] gap [3, 4] after'
        result = extract_json(text)
        assert result is not None
        assert result == [1, 2]

    def test_nested_objects_extracted_correctly(self):
        """A greedy or non-greedy regex would break on nested braces."""
        obj = {"outer": {"inner": {"deep": True}}, "count": 3}
        text = f"Analysis: {json.dumps(obj)} done."
        result = extract_json(text)
        assert result is not None
        assert result["outer"]["inner"]["deep"] is True
        assert result["count"] == 3

    def test_string_with_braces_not_confused(self):
        """Braces inside JSON string values must not affect depth counting."""
        obj = {"msg": "use {block_size} threads", "ok": True}
        text = f"Result: {json.dumps(obj)}"
        result = extract_json(text)
        assert result is not None
        assert result["msg"] == "use {block_size} threads"

    def test_string_with_escaped_quote(self):
        obj = {"name": 'say "hello"', "value": 42}
        text = f"Output: {json.dumps(obj)}"
        result = extract_json(text)
        assert result is not None
        assert result["value"] == 42

    def test_array_with_nested_objects(self):
        arr = [{"a": {"b": 1}}, {"c": 2}]
        text = f"Items: {json.dumps(arr)} end"
        result = extract_json(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["a"]["b"] == 1


# ── Balanced Extractor Tests ──────────────────────────────────

class TestExtractBalanced:

    def test_simple_object(self):
        assert _extract_balanced('x {"k": 1} y', "{", "}") == '{"k": 1}'

    def test_simple_array(self):
        assert _extract_balanced("x [1,2] y", "[", "]") == "[1,2]"

    def test_nested_object(self):
        text = '{"a": {"b": {"c": 1}}, "d": 2}'
        result = _extract_balanced(text, "{", "}")
        assert result == text

    def test_string_literal_with_close_char(self):
        text = '{"msg": "use } in config", "ok": true}'
        result = _extract_balanced(text, "{", "}")
        assert result == text

    def test_no_open_char(self):
        assert _extract_balanced("nothing here", "{", "}") is None

    def test_unbalanced_returns_none(self):
        assert _extract_balanced('{"incomplete":', "{", "}") is None

    def test_skips_earlier_noise(self):
        result = _extract_balanced('noise {"first": 1} more {"second": 2}', "{", "}")
        assert result == '{"first": 1}'

    def test_escaped_quote_in_string(self):
        text = r'{"msg": "say \"hi\"", "ok": true}'
        result = _extract_balanced(text, "{", "}")
        assert result == text


# ── Mock Provider for generate_json Tests ─────────────────────

class _MockProvider(LLMProvider):
    """In-memory mock that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def provider_name(self) -> str:
        return "mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        content = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            input_tokens=10,
            output_tokens=20,
        )


class _FailingMockProvider(LLMProvider):
    """Mock that raises an exception on generate()."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def provider_name(self) -> str:
        return "failing_mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise self._exc


class TestGenerateJson:

    @pytest.mark.asyncio
    async def test_valid_json_first_try(self):
        valid = json.dumps({"ok": True})
        provider = _MockProvider([valid])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"ok": True}
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self):
        provider = _MockProvider(["not json", '{"ok": true}'])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=3)
        assert resp.json_valid is True
        assert resp.parsed_json == {"ok": True}
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_raises_LLMResponseParseError_on_exhaustion(self):
        """All retries exhausted + parse failure = LLMResponseParseError."""
        provider = _MockProvider(["bad", "also bad", "still bad"])
        with pytest.raises(LLMResponseParseError, match="Failed to parse JSON"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)
        assert provider._call_count == 3

    @pytest.mark.asyncio
    async def test_json_in_markdown_retry_succeeds(self):
        provider = _MockProvider([
            "no json here",
            '```json\n{"recovered": true}\n```',
        ])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=2)
        assert resp.json_valid is True
        assert resp.parsed_json["recovered"] is True

    @pytest.mark.asyncio
    async def test_provider_error_propagates(self):
        """LLMError from the provider propagates without returning a response."""
        provider = _FailingMockProvider(LLMError("API connection failed"))
        with pytest.raises(LLMError, match="API connection failed"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagates(self):
        """LLMRateLimitError (subclass of LLMError) propagates correctly."""
        provider = _FailingMockProvider(LLMRateLimitError("Rate limit exceeded"))
        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)

    @pytest.mark.asyncio
    async def test_system_prompt_not_mutated(self):
        req = LLMRequest(prompt="test", system_prompt="Original prompt")
        provider = _MockProvider(["not json", '{"ok": true}'])
        await provider.generate_json(req, retries=3)
        assert req.system_prompt == "Original prompt"

    @pytest.mark.asyncio
    async def test_array_response_wrapped_in_items(self):
        """Bare array wrapped as {"items": [...]} per contract."""
        provider = _MockProvider(["[1, 2, 3]"])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"items": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_retries_zero_still_runs_once(self):
        provider = _MockProvider(['{"ok": true}'])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=0)
        assert resp.json_valid is True
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_json_values_in_response(self):
        """Provider returns text with multiple JSON objects; first valid one is extracted."""
        text = 'Analysis: {"partial": true} then {"complete": true}'
        provider = _MockProvider([text])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"partial": True}

    @pytest.mark.asyncio
    async def test_nested_json_extracted_on_first_try(self):
        obj = {"a": {"b": {"c": 1}}, "list": [1, 2]}
        provider = _MockProvider([json.dumps(obj)])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json["a"]["b"]["c"] == 1
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_parse_error_includes_response_preview(self):
        """LLMResponseParseError message contains a truncated preview of the last response."""
        long_bad = "x" * 500
        provider = _MockProvider([long_bad, long_bad])
        with pytest.raises(LLMResponseParseError) as exc_info:
            await provider.generate_json(LLMRequest(prompt="test"), retries=2)
        # The error message should contain a truncated preview (200 chars)
        assert "xxx" in str(exc_info.value)
        assert str(exc_info.value).count("xxx") > 0


# ── Prompt Builder Tests ───────────────────────────────────────

class TestPromptBuilders:

    @pytest.fixture
    def kernel(self) -> KernelSource:
        return KernelSource(
            name="matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void matmul() {}",
            entry_point="matmul",
        )

    @pytest.fixture
    def bottleneck(self) -> BottleneckResult:
        return BottleneckResult(
            primary=BottleneckClass.MEMORY_BOUND,
            confidence=0.85,
            reasoning="DRAM throughput > SM throughput",
            recommended_transforms=["shared_memory_tiling"],
        )

    @pytest.fixture
    def metrics(self) -> ProfileMetrics:
        return ProfileMetrics(
            kernel_name="matmul",
            gpu_time_us=150.0,
            sm_throughput=45.0,
            dram_throughput=82.0,
            achieved_occupancy=0.625,
            registers_per_thread=40,
            block_size=(16, 16, 1),
            occupancy_limited_by_registers=True,
        )

    def test_format_kernel_source(self, kernel):
        text = format_kernel_source(kernel)
        assert "matmul.cu" in text
        assert "cuda" in text
        assert "__global__ void matmul" in text
        assert "```cuda" in text

    def test_format_bottleneck(self, bottleneck):
        text = format_bottleneck(bottleneck)
        assert "memory_bound" in text
        assert "85%" in text
        assert "DRAM" in text
        assert "shared_memory_tiling" in text

    def test_format_metrics(self, metrics):
        text = format_metrics(metrics)
        assert "150.00" in text
        assert "82.0%" in text
        assert "62.5%" in text
        assert "40" in text
        assert "registers" in text.lower()

    def test_build_transform_prompt_structure(self, kernel, bottleneck, metrics):
        request = build_transform_prompt(kernel, bottleneck, metrics)

        assert request.json_mode is True
        assert len(request.system_prompt) > 100
        assert "shared_memory_tiling" in request.system_prompt
        assert "loop_unrolling" in request.system_prompt
        assert "matmul.cu" in request.prompt
        assert "memory_bound" in request.prompt
        assert "82.0%" in request.prompt

    def test_build_transform_prompt_without_metrics(self, kernel, bottleneck):
        request = build_transform_prompt(kernel, bottleneck)
        assert "matmul.cu" in request.prompt
        assert request.json_mode is True

    def test_variant_prompt_includes_history(self, kernel, bottleneck):
        variant = KernelVariant(
            variant_id="v1_tiled",
            parent_source=kernel,
            source_code="// optimized",
            transform_history=[
                __import__(
                    "cuda_opt_agent.schemas.transform", fromlist=["TransformSpec"]
                ).TransformSpec(
                    transform_name="shared_memory_tiling",
                    params={"tile_size": 16},
                    reason="memory bound",
                ),
            ],
        )
        request = build_transform_prompt(variant, bottleneck)
        assert "v1_tiled" in request.prompt
        assert "shared_memory_tiling" in request.prompt
        assert "tile_size" in request.prompt


# ── Factory Tests ──────────────────────────────────────────────

class TestFactory:

    def test_unknown_provider_raises(self):
        config = LLMConfig(provider="nonexistent")  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            create_provider(config)

    def test_claude_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(provider="claude")
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            create_provider(config)

    def test_openai_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig(provider="openai")
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            create_provider(config)

    def test_gemini_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = LLMConfig(provider="gemini")
        with pytest.raises(LLMError, match="GEMINI_API_KEY"):
            create_provider(config)

    def test_ollama_creation(self):
        config = LLMConfig(provider="ollama")
        provider = create_provider(config)
        assert provider.provider_name() == "ollama"
        assert provider.estimate_cost(1000, 1000) == 0.0
"""Tests for LLM layer: provider ABC, JSON extraction, prompts, factory."""

from __future__ import annotations

import json

import pytest

from cuda_opt_agent.core.exceptions import (
    ConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
)
from cuda_opt_agent.llm.provider import LLMProvider, extract_json, _extract_balanced
from cuda_opt_agent.llm.factory import create_provider
from cuda_opt_agent.core.config import LLMConfig, LLMProviderConfig
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics
from cuda_opt_agent.llm.prompts.transform import build_transform_prompt
from cuda_opt_agent.llm.prompts.base import (
    format_bottleneck,
    format_kernel_source,
    format_metrics,
)


# ── JSON Extraction Tests ──────────────────────────────────────

class TestExtractJson:

    def test_direct_json_object(self):
        text = '{"transforms": [], "analysis_summary": "ok"}'
        result = extract_json(text)
        assert result is not None
        assert "transforms" in result

    def test_direct_json_array(self):
        text = '[1, 2, 3]'
        result = extract_json(text)
        assert result is not None
        assert result == [1, 2, 3]

    def test_json_in_markdown_fence(self):
        text = 'Here is the analysis:\n```json\n{"ok": true}\n```\nDone.'
        result = extract_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_json_in_plain_fence(self):
        text = '```{"ok": true}\n```'
        result = extract_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_json_embedded_in_text(self):
        text = 'Based on analysis, the result is {"speedup": 2.5} which is good.'
        result = extract_json(text)
        assert result is not None
        assert result["speedup"] == 2.5

    def test_json_array_embedded_in_text(self):
        text = 'The results are [10, 20, 30] as expected.'
        result = extract_json(text)
        assert result is not None
        assert result == [10, 20, 30]

    def test_json_array_in_fence(self):
        text = 'Here are the items:\n```json\n["a", "b", "c"]\n```'
        result = extract_json(text)
        assert result is not None
        assert result == ["a", "b", "c"]

    def test_empty_string(self):
        assert extract_json("") is None
        assert extract_json("  ") is None

    def test_no_json(self):
        assert extract_json("This is just plain text with no JSON.") is None

    def test_invalid_json(self):
        assert extract_json("{not valid json}") is None

    def test_nested_json(self):
        obj = {
            "transforms": [
                {"name": "tiling", "params": {"tile_size": 16}},
                {"name": "unrolling", "params": {"factor": 4}},
            ],
            "warnings": ["may increase register pressure"],
        }
        text = f"Analysis complete:\n```json\n{json.dumps(obj)}\n```"
        result = extract_json(text)
        assert result is not None
        assert len(result["transforms"]) == 2
        assert result["transforms"][0]["params"]["tile_size"] == 16

    def test_prefers_direct_over_fence(self):
        text = '{"direct": true}'
        result = extract_json(text)
        assert result is not None
        assert result["direct"] is True

    def test_handles_whitespace_around_fence(self):
        text = '  \n  ```json\n{"padded": true}\n```  \n  '
        result = extract_json(text)
        assert result is not None
        assert result["padded"] is True

    def test_multiple_objects_picks_first(self):
        """Greedy regex would over-match. Balanced scanner picks the first complete value."""
        text = 'prefix {"a": 1} middle {"b": 2} suffix'
        result = extract_json(text)
        assert result is not None
        assert result == {"a": 1}

    def test_multiple_arrays_picks_first(self):
        text = 'before [1, 2] gap [3, 4] after'
        result = extract_json(text)
        assert result is not None
        assert result == [1, 2]

    def test_nested_objects_extracted_correctly(self):
        """A greedy or non-greedy regex would break on nested braces."""
        obj = {"outer": {"inner": {"deep": True}}, "count": 3}
        text = f"Analysis: {json.dumps(obj)} done."
        result = extract_json(text)
        assert result is not None
        assert result["outer"]["inner"]["deep"] is True
        assert result["count"] == 3

    def test_string_with_braces_not_confused(self):
        """Braces inside JSON string values must not affect depth counting."""
        obj = {"msg": "use {block_size} threads", "ok": True}
        text = f"Result: {json.dumps(obj)}"
        result = extract_json(text)
        assert result is not None
        assert result["msg"] == "use {block_size} threads"

    def test_string_with_escaped_quote(self):
        obj = {"name": 'say "hello"', "value": 42}
        text = f"Output: {json.dumps(obj)}"
        result = extract_json(text)
        assert result is not None
        assert result["value"] == 42

    def test_array_with_nested_objects(self):
        arr = [{"a": {"b": 1}}, {"c": 2}]
        text = f"Items: {json.dumps(arr)} end"
        result = extract_json(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["a"]["b"] == 1


# ── Balanced Extractor Tests ──────────────────────────────────

class TestExtractBalanced:

    def test_simple_object(self):
        assert _extract_balanced('x {"k": 1} y', "{", "}") == '{"k": 1}'

    def test_simple_array(self):
        assert _extract_balanced("x [1,2] y", "[", "]") == "[1,2]"

    def test_nested_object(self):
        text = '{"a": {"b": {"c": 1}}, "d": 2}'
        result = _extract_balanced(text, "{", "}")
        assert result == text

    def test_string_literal_with_close_char(self):
        text = '{"msg": "use } in config", "ok": true}'
        result = _extract_balanced(text, "{", "}")
        assert result == text

    def test_no_open_char(self):
        assert _extract_balanced("nothing here", "{", "}") is None

    def test_unbalanced_returns_none(self):
        assert _extract_balanced('{"incomplete":', "{", "}") is None

    def test_skips_earlier_noise(self):
        result = _extract_balanced('noise {"first": 1} more {"second": 2}', "{", "}")
        assert result == '{"first": 1}'

    def test_escaped_quote_in_string(self):
        text = r'{"msg": "say \"hi\"", "ok": true}'
        result = _extract_balanced(text, "{", "}")
        assert result == text


# ── Mock Provider for generate_json Tests ─────────────────────

class _MockProvider(LLMProvider):
    """In-memory mock that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def provider_name(self) -> str:
        return "mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        content = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            input_tokens=10,
            output_tokens=20,
        )


class _FailingMockProvider(LLMProvider):
    """Mock that raises an exception on generate()."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def provider_name(self) -> str:
        return "failing_mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise self._exc


class TestGenerateJson:

    @pytest.mark.asyncio
    async def test_valid_json_first_try(self):
        valid = json.dumps({"ok": True})
        provider = _MockProvider([valid])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"ok": True}
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self):
        provider = _MockProvider(["not json", '{"ok": true}'])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=3)
        assert resp.json_valid is True
        assert resp.parsed_json == {"ok": True}
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_raises_LLMResponseParseError_on_exhaustion(self):
        """All retries exhausted + parse failure = LLMResponseParseError."""
        provider = _MockProvider(["bad", "also bad", "still bad"])
        with pytest.raises(LLMResponseParseError, match="Failed to parse JSON"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)
        assert provider._call_count == 3

    @pytest.mark.asyncio
    async def test_json_in_markdown_retry_succeeds(self):
        provider = _MockProvider([
            "no json here",
            '```json\n{"recovered": true}\n```',
        ])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=2)
        assert resp.json_valid is True
        assert resp.parsed_json["recovered"] is True

    @pytest.mark.asyncio
    async def test_provider_error_propagates(self):
        """LLMError from the provider propagates without returning a response."""
        provider = _FailingMockProvider(LLMError("API connection failed"))
        with pytest.raises(LLMError, match="API connection failed"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagates(self):
        """LLMRateLimitError (subclass of LLMError) propagates correctly."""
        provider = _FailingMockProvider(LLMRateLimitError("Rate limit exceeded"))
        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)

    @pytest.mark.asyncio
    async def test_system_prompt_not_mutated(self):
        req = LLMRequest(prompt="test", system_prompt="Original prompt")
        provider = _MockProvider(["not json", '{"ok": true}'])
        await provider.generate_json(req, retries=3)
        assert req.system_prompt == "Original prompt"

    @pytest.mark.asyncio
    async def test_array_response_wrapped_in_items(self):
        """Bare array wrapped as {"items": [...]} per contract."""
        provider = _MockProvider(["[1, 2, 3]"])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"items": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_retries_zero_still_runs_once(self):
        provider = _MockProvider(['{"ok": true}'])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=0)
        assert resp.json_valid is True
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_json_values_in_response(self):
        """Provider returns text with multiple JSON objects; first valid one is extracted."""
        text = 'Analysis: {"partial": true} then {"complete": true}'
        provider = _MockProvider([text])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"partial": True}

    @pytest.mark.asyncio
    async def test_nested_json_extracted_on_first_try(self):
        obj = {"a": {"b": {"c": 1}}, "list": [1, 2]}
        provider = _MockProvider([json.dumps(obj)])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json["a"]["b"]["c"] == 1
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_parse_error_includes_response_preview(self):
        """LLMResponseParseError message contains a truncated preview of the last response."""
        long_bad = "x" * 500
        provider = _MockProvider([long_bad, long_bad])
        with pytest.raises(LLMResponseParseError) as exc_info:
            await provider.generate_json(LLMRequest(prompt="test"), retries=2)
        # The error message should contain a truncated preview (200 chars)
        assert "xxx" in str(exc_info.value)
        assert str(exc_info.value).count("xxx") > 0


# ── Prompt Builder Tests ───────────────────────────────────────

class TestPromptBuilders:

    @pytest.fixture
    def kernel(self) -> KernelSource:
        return KernelSource(
            name="matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void matmul() {}",
            entry_point="matmul",
        )

    @pytest.fixture
    def bottleneck(self) -> BottleneckResult:
        return BottleneckResult(
            primary=BottleneckClass.MEMORY_BOUND,
            confidence=0.85,
            reasoning="DRAM throughput > SM throughput",
            recommended_transforms=["shared_memory_tiling"],
        )

    @pytest.fixture
    def metrics(self) -> ProfileMetrics:
        return ProfileMetrics(
            kernel_name="matmul",
            gpu_time_us=150.0,
            sm_throughput=45.0,
            dram_throughput=82.0,
            achieved_occupancy=0.625,
            registers_per_thread=40,
            block_size=(16, 16, 1),
            occupancy_limited_by_registers=True,
        )

    def test_format_kernel_source(self, kernel):
        text = format_kernel_source(kernel)
        assert "matmul.cu" in text
        assert "cuda" in text
        assert "__global__ void matmul" in text
        assert "```cuda" in text

    def test_format_bottleneck(self, bottleneck):
        text = format_bottleneck(bottleneck)
        assert "memory_bound" in text
        assert "85%" in text
        assert "DRAM" in text
        assert "shared_memory_tiling" in text

    def test_format_metrics(self, metrics):
        text = format_metrics(metrics)
        assert "150.00" in text
        assert "82.0%" in text
        assert "62.5%" in text
        assert "40" in text
        assert "registers" in text.lower()

    def test_build_transform_prompt_structure(self, kernel, bottleneck, metrics):
        request = build_transform_prompt(kernel, bottleneck, metrics)

        assert request.json_mode is True
        assert len(request.system_prompt) > 100
        assert "shared_memory_tiling" in request.system_prompt
        assert "loop_unrolling" in request.system_prompt
        assert "matmul.cu" in request.prompt
        assert "memory_bound" in request.prompt
        assert "82.0%" in request.prompt

    def test_build_transform_prompt_without_metrics(self, kernel, bottleneck):
        request = build_transform_prompt(kernel, bottleneck)
        assert "matmul.cu" in request.prompt
        assert request.json_mode is True

    def test_variant_prompt_includes_history(self, kernel, bottleneck):
        variant = KernelVariant(
            variant_id="v1_tiled",
            parent_source=kernel,
            source_code="// optimized",
            transform_history=[
                __import__(
                    "cuda_opt_agent.schemas.transform", fromlist=["TransformSpec"]
                ).TransformSpec(
                    transform_name="shared_memory_tiling",
                    params={"tile_size": 16},
                    reason="memory bound",
                ),
            ],
        )
        request = build_transform_prompt(variant, bottleneck)
        assert "v1_tiled" in request.prompt
        assert "shared_memory_tiling" in request.prompt
        assert "tile_size" in request.prompt


# ── Factory Tests ──────────────────────────────────────────────

class TestFactory:

    def test_unknown_provider_raises(self):
        config = LLMConfig(provider="nonexistent")  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            create_provider(config)

    def test_claude_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(provider="claude")
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            create_provider(config)

    def test_openai_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig(provider="openai")
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            create_provider(config)

    def test_gemini_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = LLMConfig(provider="gemini")
        with pytest.raises(LLMError, match="GEMINI_API_KEY"):
            create_provider(config)

    def test_ollama_creation(self):
        config = LLMConfig(provider="ollama")
        provider = create_provider(config)
        assert provider.provider_name() == "ollama"
        assert provider.estimate_cost(1000, 1000) == 0.0


CHANGELOG_LAYER2.md

markdown
# CHANGELOG — Layer 2 (LLM Providers)

## Fixed

### Critical: Greedy regex over-matched in JSON extraction
- **File:** `llm/provider.py`
- **Bug:** Strategy #3 used `r"\{.*\}"` and `r"$$.*$$"` (greedy regexes).
  On input like `{"a":1} junk {"b":2}`, the pattern matched the entire span
  `{"a":1} junk {"b":2}`, which is invalid JSON. Nested structures like
  `{"a": {"b": 1}}` also failed — a non-greedy `.*?` variant would stop at
  the *inner* `}`, producing the truncated `{"a": {"b": 1}`.
- **Fix:** Replaced regex-based extraction with `_extract_balanced()`, a
  character-level scanner that tracks brace/bracket nesting depth and respects
  JSON string literals (including `\"` escapes). Removed the now-unused
  `_JSON_OBJECT_RE` and `_JSON_ARRAY_RE` constants.
- **Impact:** Strategy #3 now correctly extracts the first complete JSON value
  regardless of nesting depth, multiple occurrences, or braces inside strings.

### Ambiguous contract: `LLMResponseParseError` imported but never raised
- **File:** `llm/provider.py`
- **Bug:** On exhaustion of all retries, `generate_json()` returned the last
  response with `json_valid = False` instead of raising the already-defined
  `LLMResponseParseError`. This made the contract ambiguous: callers had to
  check a boolean flag rather than catching an exception.
- **Fix:** `generate_json()` now raises `LLMResponseParseError` with a
  truncated preview of the last response content. The contract is now:
  **return a successfully parsed response, or raise**.
- **Impact:** Callers must catch `LLMResponseParseError` (subclass of `LLMError`)
  for parse failures. Existing callers that catch `LLMError` will still work
  since `LLMResponseParseError` inherits from it.

### Array wrapping contract documented
- **File:** `llm/provider.py`
- **Status:** The `{"items": array}` wrapping for bare JSON arrays is retained
  as a defensive fallback. Added docstring and inline comment documenting that
  `TRANSFORM_RESPONSE_SCHEMA` always expects a dict, so this branch should
  never trigger in normal operation.

### JSON array regex (carried from prior pass)
- **File:** `llm/provider.py`
- **Status:** `_JSON_ARRAY_RE` was previously `$$.*$$` (matching literal `$`).
  This is now moot — the constant has been removed in favor of
  `_extract_balanced()`.

### System prompt accumulation on retry (carried from prior pass)
- **File:** `llm/provider.py`
- **Status:** Unchanged from prior fix (stores `original_system` before loop).

## Added

### New internal function: `_extract_balanced()`
- Character-level balanced delimiter scanner
- Correctly handles: nesting, string literals, escaped quotes, multiple values

### Test coverage expansion (`tests/unit/test_llm.py`)

**`TestExtractJson` (JSON extraction):**
- `test_multiple_objects_picks_first` — validates greedy-regex fix
- `test_multiple_arrays_picks_first` — same for arrays
- `test_nested_objects_extracted_correctly` — nested braces parse correctly
- `test_string_with_braces_not_confused` — braces in string values ignored
- `test_string_with_escaped_quote` — `\"` in strings handled
- `test_array_with_nested_objects` — arrays containing nested objects

**`TestExtractBalanced` (new test class, 8 tests):**
- `test_simple_object` / `test_simple_array`
- `test_nested_object` — 3 levels deep
- `test_string_literal_with_close_char` — `}` inside string
- `test_no_open_char` — returns None
- `test_unbalanced_returns_none` — unclosed delimiter
- `test_skips_earlier_noise` — finds first balanced match past noise
- `test_escaped_quote_in_string` — `\"` not misinterpreted

**`TestGenerateJson` (retry + error handling):**
- `test_raises_LLMResponseParseError_on_exhaustion` — replaces former
  `test_returns_last_response_on_exhaustion`; now expects exception
- `test_provider_error_propagates` — LLMError from generate() bubbles up
- `test_rate_limit_error_propagates` — LLMRateLimitError bubbles up
- `test_system_prompt_not_mutated` — original request unmodified
- `test_array_response_wrapped_in_items` — wrapping contract
- `test_retries_zero_still_runs_once` — edge case
- `test_multiple_json_values_in_response` — first valid value wins
- `test_nested_json_extracted_on_first_try` — nested on first attempt
- `test_parse_error_includes_response_preview` — error message content

### New mock: `_FailingMockProvider`
- Raises a configurable exception on every `generate()` call
- Used to test exception propagation through `generate_json()`

## Not Changed (by design)

- Concrete providers (`claude_provider.py`, `openai_provider.py`,
  `gemini_provider.py`, `ollama_provider.py`) — no bugs found.
- Factory (`factory.py`) — match/case dispatch correct.
- Prompt builders (`prompts/`) — no bugs found.
- `__init__.py` exports — unchanged.
- Public interfaces preserved; `generate_json()` return type on success is
  unchanged; only the failure path now raises instead of returning.
# CHANGELOG — Layer 2 (LLM Providers)

## Fixed

### Critical: Greedy regex over-matched in JSON extraction
- **File:** `llm/provider.py`
- **Bug:** Strategy #3 used `r"\{.*\}"` and `r"$$.*$$"` (greedy regexes).
  On input like `{"a":1} junk {"b":2}`, the pattern matched the entire span
  `{"a":1} junk {"b":2}`, which is invalid JSON. Nested structures like
  `{"a": {"b": 1}}` also failed — a non-greedy `.*?` variant would stop at
  the *inner* `}`, producing the truncated `{"a": {"b": 1}`.
- **Fix:** Replaced regex-based extraction with `_extract_balanced()`, a
  character-level scanner that tracks brace/bracket nesting depth and respects
  JSON string literals (including `\"` escapes). Removed the now-unused
  `_JSON_OBJECT_RE` and `_JSON_ARRAY_RE` constants.
- **Impact:** Strategy #3 now correctly extracts the first complete JSON value
  regardless of nesting depth, multiple occurrences, or braces inside strings.

### Ambiguous contract: `LLMResponseParseError` imported but never raised
- **File:** `llm/provider.py`
- **Bug:** On exhaustion of all retries, `generate_json()` returned the last
  response with `json_valid = False` instead of raising the already-defined
  `LLMResponseParseError`. This made the contract ambiguous: callers had to
  check a boolean flag rather than catching an exception.
- **Fix:** `generate_json()` now raises `LLMResponseParseError` with a
  truncated preview of the last response content. The contract is now:
  **return a successfully parsed response, or raise**.
- **Impact:** Callers must catch `LLMResponseParseError` (subclass of `LLMError`)
  for parse failures. Existing callers that catch `LLMError` will still work
  since `LLMResponseParseError` inherits from it.

### Array wrapping contract documented
- **File:** `llm/provider.py`
- **Status:** The `{"items": array}` wrapping for bare JSON arrays is retained
  as a defensive fallback. Added docstring and inline comment documenting that
  `TRANSFORM_RESPONSE_SCHEMA` always expects a dict, so this branch should
  never trigger in normal operation.

### JSON array regex (carried from prior pass)
- **File:** `llm/provider.py`
- **Status:** `_JSON_ARRAY_RE` was previously `$$.*$$` (matching literal `$`).
  This is now moot — the constant has been removed in favor of
  `_extract_balanced()`.

### System prompt accumulation on retry (carried from prior pass)
- **File:** `llm/provider.py`
- **Status:** Unchanged from prior fix (stores `original_system` before loop).

## Added

### New internal function: `_extract_balanced()`
- Character-level balanced delimiter scanner
- Correctly handles: nesting, string literals, escaped quotes, multiple values

### Test coverage expansion (`tests/unit/test_llm.py`)

**`TestExtractJson` (JSON extraction):**
- `test_multiple_objects_picks_first` — validates greedy-regex fix
- `test_multiple_arrays_picks_first` — same for arrays
- `test_nested_objects_extracted_correctly` — nested braces parse correctly
- `test_string_with_braces_not_confused` — braces in string values ignored
- `test_string_with_escaped_quote` — `\"` in strings handled
- `test_array_with_nested_objects` — arrays containing nested objects

**`TestExtractBalanced` (new test class, 8 tests):**
- `test_simple_object` / `test_simple_array`
- `test_nested_object` — 3 levels deep
- `test_string_literal_with_close_char` — `}` inside string
- `test_no_open_char` — returns None
- `test_unbalanced_returns_none` — unclosed delimiter
- `test_skips_earlier_noise` — finds first balanced match past noise
- `test_escaped_quote_in_string` — `\"` not misinterpreted

**`TestGenerateJson` (retry + error handling):**
- `test_raises_LLMResponseParseError_on_exhaustion` — replaces former
  `test_returns_last_response_on_exhaustion`; now expects exception
- `test_provider_error_propagates` — LLMError from generate() bubbles up
- `test_rate_limit_error_propagates` — LLMRateLimitError bubbles up
- `test_system_prompt_not_mutated` — original request unmodified
- `test_array_response_wrapped_in_items` — wrapping contract
- `test_retries_zero_still_runs_once` — edge case
- `test_multiple_json_values_in_response` — first valid value wins
- `test_nested_json_extracted_on_first_try` — nested on first attempt
- `test_parse_error_includes_response_preview` — error message content

### New mock: `_FailingMockProvider`
- Raises a configurable exception on every `generate()` call
- Used to test exception propagation through `generate_json()`

## Not Changed (by design)

- Concrete providers (`claude_provider.py`, `openai_provider.py`,
  `gemini_provider.py`, `ollama_provider.py`) — no bugs found.
- Factory (`factory.py`) — match/case dispatch correct.
- Prompt builders (`prompts/`) — no bugs found.
- `__init__.py` exports — unchanged.
- Public interfaces preserved; `generate_json()` return type on success is
  unchanged; only the failure path now raises instead of returning.


Summary

Fix	What changed	Risk
Greedy regex → _extract_balanced()	Character-level scanner replaces regex for strategy #3	Low — pure improvement; strategy #1 and #2 unchanged
LLMResponseParseError now raised	Failure path raises instead of returning	Medium — callers that checked json_valid after exhaustion must now catch the exception. Since LLMResponseParseError is a subclass of LLMError, any existing except LLMError handler still works.
Array wrapping documented	Docstring + inline comment
