"""Mistral La Plateforme wrapper — the single choke point for every LLM call.

Phase 2.2. All generation/judge/decomposition calls go through ``MistralClient`` so
retries, backoff, and (later) Langfuse tracing live in exactly one place.

Rate limits are the binding constraint on the free tier (RPM, not tokens), so the
client does **mandatory retry with exponential backoff on 429**. Transient 5xx
errors are retried too; everything else is raised immediately.

Testability: the low-level HTTP call (``transport``) and ``sleep`` are injectable, so
the retry/backoff logic is exercised in unit tests without a network or real waits.
The default transport uses the pinned ``mistralai`` SDK.
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable, Optional

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class MistralError(RuntimeError):
    """Raised when a call ultimately fails (after exhausting retries, or non-retryable)."""


def _status_of(exc: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an SDK exception."""
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Fall back to sniffing the message (e.g. "... 429 Too Many Requests ...").
    msg = str(exc)
    for code in _RETRYABLE_STATUS:
        if str(code) in msg:
            return code
    if "rate limit" in msg.lower():
        return 429
    return None


class MistralClient:
    """Config-driven Mistral chat client with centralized retry/backoff."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 6,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 60.0,
        transport: Optional[Callable[..., str]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        jitter: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self._transport = transport
        self._sleep = sleep_fn
        self._jitter = jitter
        self._sdk = None  # lazily created real SDK client (only when no transport injected)

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        *,
        use_final_model: bool = False,
        transport: Optional[Callable[..., str]] = None,
        **overrides,
    ) -> "MistralClient":
        """Build from the ``generation`` block of config.

        Defaults to the dev model (Mistral Small); ``use_final_model=True`` selects
        the large model for the final Phase 3 run. API key comes from the
        MISTRAL_API_KEY environment variable (loaded from .env).
        """
        g = cfg["generation"]
        model = g["model_final"] if use_final_model else g["model_dev"]
        params = dict(
            model=model,
            api_key=os.environ.get("MISTRAL_API_KEY"),
            temperature=g.get("temperature", 0.0),
            max_tokens=g.get("max_tokens", 1024),
            max_retries=g.get("max_retries", 6),
            backoff_base_seconds=g.get("backoff_base_seconds", 2.0),
            backoff_max_seconds=g.get("backoff_max_seconds", 60.0),
            transport=transport,
        )
        params.update(overrides)
        return cls(**params)

    # -- backoff --------------------------------------------------------------

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff for a 0-based attempt index, capped, with optional jitter."""
        delay = min(self.backoff_max_seconds, self.backoff_base_seconds * (2 ** attempt))
        if self._jitter:
            delay += random.uniform(0, self.backoff_base_seconds)
        return min(delay, self.backoff_max_seconds)

    # -- transport ------------------------------------------------------------

    def _raw_call(self, messages, temperature, max_tokens, response_format) -> str:
        if self._transport is not None:
            return self._transport(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        return self._sdk_call(messages, temperature, max_tokens, response_format)

    def _sdk_call(self, messages, temperature, max_tokens, response_format) -> str:
        from mistralai import Mistral  # local import: heavy, network-bound

        if self._sdk is None:
            if not self.api_key:
                raise MistralError("MISTRAL_API_KEY is not set (put it in .env).")
            self._sdk = Mistral(api_key=self.api_key)

        kwargs = dict(model=self.model, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._sdk.chat.complete(**kwargs)
        return resp.choices[0].message.content

    # -- public API -----------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Run a chat completion, retrying on 429/5xx with exponential backoff.

        ``messages`` is a list of {"role", "content"} dicts. ``json_mode=True`` asks
        Mistral for a JSON object response. Returns the assistant message text.
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        response_format = {"type": "json_object"} if json_mode else None

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._raw_call(messages, temperature, max_tokens, response_format)
            except Exception as exc:  # noqa: BLE001 - classify below
                status = _status_of(exc)
                retryable = status in _RETRYABLE_STATUS
                if not retryable or attempt == self.max_retries:
                    raise MistralError(
                        f"Mistral call failed (status={status}, attempt={attempt + 1}"
                        f"/{self.max_retries + 1}): {exc}"
                    ) from exc
                last_exc = exc
                self._sleep(self._backoff_seconds(attempt))
        # Unreachable, but keeps type-checkers happy.
        raise MistralError(f"Mistral call failed: {last_exc}")
