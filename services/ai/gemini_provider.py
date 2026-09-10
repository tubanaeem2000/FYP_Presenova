"""
Gemini Provider — AIProvider implementation for Google Gemini.

Supports both the current google.genai SDK and the legacy google.generativeai SDK.
Implements retry logic, model fallback chain, and timeout handling.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

from dotenv import load_dotenv

from services.ai.base_provider import AIProvider

load_dotenv()
logger = logging.getLogger(__name__)

# ── SDK Imports (graceful degradation) ──────────────────────────────────────
try:
    import google.genai as current_genai
except ImportError:
    current_genai = None

try:
    import google.generativeai as legacy_genai
except ImportError:
    legacy_genai = None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _clean_json_response(raw: str) -> str:
    """Remove markdown fences and return the first valid JSON value."""
    if not isinstance(raw, str):
        return ''
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in '{[':
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index:index + end].strip()
        except json.JSONDecodeError:
            continue
    return text


class GeminiProvider(AIProvider):
    """Gemini AI provider with retry, fallback, and timeout support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', '').strip()
        self.model_name = model or os.getenv('GEMINI_MODEL', 'gemini-1.5-flash').strip()
        self.fallback_models = fallback_models or [
            m.strip() for m in os.getenv('GEMINI_FALLBACK_MODELS', 'gemini-2.0-flash').split(',') if m.strip()
        ]
        self.timeout_seconds = max(1.0, timeout_seconds or _env_float('GEMINI_TIMEOUT_SECONDS', 45.0))
        self.max_retries = max(1, max_retries or _env_int('GEMINI_MAX_RETRIES', 2))
        self.model_candidates = [
            self.model_name,
            *self.fallback_models,
        ]

        self._new_client = None
        self._legacy_configured = False
        self._available = False
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize Gemini SDK clients (current and/or legacy)."""
        offline = _env_bool('PRESENTATION_REWRITER_OFFLINE', False)
        if offline or not self.api_key:
            logger.info("[gemini_provider] Gemini disabled (offline or no API key).")
            return

        if current_genai is not None:
            try:
                self._new_client = current_genai.Client(api_key=self.api_key)
                logger.info("[gemini_provider] Current Gemini SDK initialized.")
            except Exception as exc:
                logger.warning("[gemini_provider] Current SDK failed: %s", exc)

        if self._new_client is None and legacy_genai is not None:
            try:
                legacy_genai.configure(api_key=self.api_key, transport='rest')
                self._legacy_configured = True
                logger.info("[gemini_provider] Legacy Gemini SDK initialized.")
            except Exception as exc:
                logger.warning("[gemini_provider] Legacy SDK failed: %s", exc)

        self._available = bool(
            self.api_key and not offline and (self._new_client is not None or self._legacy_configured)
        )
        if not self._available:
            logger.info("[gemini_provider] Gemini unavailable; local fallback active.")

    # ── AIProvider interface ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available

    def get_model_info(self) -> dict:
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "fallback_models": self.fallback_models,
            "available": self._available,
            "sdk": "current" if self._new_client else ("legacy" if self._legacy_configured else "none"),
        }

    def count_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4 if text else 0

    def get_capabilities(self) -> dict:
        return {
            "supports_structured": True,
            "supports_streaming": False,
            "supports_system_instruction": True,
            "max_context_tokens": 128000,
            "max_output_tokens": 65536,
        }

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate text with retry and model fallback."""
        if not self._available:
            logger.warning("[gemini_provider] Provider not available; returning empty.")
            return ''

        last_error = None
        for attempt in range(self.max_retries + 1):
            for model_name in self.model_candidates:
                try:
                    result = self._call_model(
                        model_name=model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        system_instruction=system_instruction,
                    )
                    if result:
                        return result
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "[gemini_provider] Attempt %d with %s failed: %s",
                        attempt + 1, model_name, exc,
                    )
                    continue

            if attempt < self.max_retries:
                wait = 2.0 ** attempt
                logger.info("[gemini_provider] Retrying in %.1fs...", wait)
                time.sleep(wait)

        logger.error("[gemini_provider] All attempts exhausted: %s", last_error)
        return ''

    def generate_structured(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
    ) -> dict | list:
        """Generate structured JSON output."""
        raw = self.generate(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )
        if not raw:
            return {}

        cleaned = _clean_json_response(raw)
        if not cleaned:
            logger.warning("[gemini_provider] No JSON found in response.")
            return {}

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("[gemini_provider] JSON parse failed: %s", exc)
            return {}

    # ── Internal model call ───────────────────────────────────────────────────

    def _call_model(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Call a specific Gemini model with timeout."""
        if self._new_client:
            return self._call_current_sdk(
                model_name, prompt, temperature, max_output_tokens, system_instruction
            )
        elif self._legacy_configured:
            return self._call_legacy_sdk(
                model_name, prompt, temperature, max_output_tokens, system_instruction
            )
        raise RuntimeError("No Gemini SDK available.")

    def _call_current_sdk(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Use the current google.genai SDK."""
        contents = [] if not system_instruction else []
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._new_client.models.generate_content,
                model=f"models/{model_name}",
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                } if not system_instruction else {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "system_instruction": system_instruction,
                },
            )
            try:
                response = future.result(timeout=self.timeout_seconds)
                return response.text if hasattr(response, 'text') else str(response)
            except FutureTimeout:
                raise TimeoutError(f"Gemini {model_name} timed out after {self.timeout_seconds}s.")

    def _call_legacy_sdk(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Use the legacy google.generativeai SDK."""
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        kwargs = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        model = legacy_genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            **kwargs,
        )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(model.generate_content, prompt)
            try:
                response = future.result(timeout=self.timeout_seconds)
                return response.text if hasattr(response, 'text') else str(response)
            except FutureTimeout:
                raise TimeoutError(f"Gemini {model_name} timed out after {self.timeout_seconds}s.")

