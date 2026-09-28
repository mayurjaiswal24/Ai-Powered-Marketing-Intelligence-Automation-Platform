"""Gemini client (official google-genai SDK) with plain-English, typed errors.

- The model comes from GEMINI_MODEL; free-tier limits are never hard-coded.
- The SDK's own automatic retries are switched OFF: a quota error (HTTP 429) must never be
  retried automatically, and any retry is a deliberate, budgeted decision (see ai/service.py).
- The API key is read from settings and never printed, logged or shown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from config.settings import AI_TEMPERATURE, AI_TIMEOUT_SECONDS


logger = logging.getLogger("marketing_intelligence.ai")


class AIError(Exception):
    """An AI problem. `kind` is machine-readable; `user_message` is safe to show."""

    def __init__(self, kind: str, user_message: str):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message


MESSAGES = {
    "disabled": "AI insights are turned off in this app. Everything else works without AI.",
    "missing_key": "No Gemini API key is set. Add GEMINI_API_KEY to the .env file (or Streamlit "
                   "secrets) and restart the app.",
    "missing_model": "No Gemini model is set. Add GEMINI_MODEL to the .env file (a model available "
                     "on your plan in Google AI Studio) and restart the app.",
    "invalid_key": "Gemini rejected the API key. Check GEMINI_API_KEY in the .env file.",
    "model_not_found": "The model named in GEMINI_MODEL was not found or is not available to this "
                       "key. Check the model name in Google AI Studio.",
    "quota": "The Gemini usage limit has been reached (too many requests). Please try again later; "
             "no retry was attempted.",
    "network": "Gemini could not be reached (network problem or timeout). Please check the "
               "connection and try again later.",
    "server": "The Gemini service is temporarily unavailable. Please try again later.",
    "invalid_response": "Gemini's answer could not be read in the expected format, so it was not "
                        "shown. Please try again later.",
    "budget": "The AI call limit for this run has been reached, so no further call was made.",
    "unknown": "The AI insights could not be generated. The rest of the dashboard is unaffected.",
}


def ai_error(kind: str) -> AIError:
    return AIError(kind, MESSAGES[kind])


@dataclass
class AIResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


def classify_error(exc: Exception) -> AIError:
    """Map SDK / network exceptions to a typed AIError (never exposing technical details)."""
    if isinstance(exc, AIError):
        return exc
    try:
        from google.genai import errors as genai_errors
        import httpx
    except ImportError:  # pragma: no cover
        return ai_error("unknown")
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        text = f"{getattr(exc, 'status', '')} {getattr(exc, 'message', '')}".lower()
        # Technical detail goes to the server log only (never the UI); Google's error messages
        # do not contain the API key.
        logger.warning("Gemini API error: code=%s status=%s message=%s", code,
                       getattr(exc, "status", ""), getattr(exc, "message", ""))
        if code == 429 or "resource_exhausted" in text or "quota" in text:
            return ai_error("quota")
        if code in (401, 403) or "api key" in text or "permission" in text:
            return ai_error("invalid_key")
        if code == 404 or "not found" in text:
            return ai_error("model_not_found")
        if code == 400 and "key" in text:
            return ai_error("invalid_key")
        if code and code >= 500:
            return ai_error("server")
        return ai_error("unknown")
    logger.warning("Gemini call failed: %s", type(exc).__name__)
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, TimeoutError, ConnectionError)):
        return ai_error("network")
    return ai_error("unknown")


class GeminiClient:
    """Thin wrapper around google-genai for one structured JSON generation."""

    def __init__(self, api_key: str, model: str, timeout_s: int = AI_TIMEOUT_SECONDS,
                 fallback_model: str = ""):
        if not api_key:
            raise ai_error("missing_key")
        if not model:
            raise ai_error("missing_model")
        # Used once, automatically, when the main model is overloaded ("high demand", HTTP 503).
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else ""
        from google import genai
        from google.genai import types
        self._types = types
        self.model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_s * 1000,
                                           retry_options=types.HttpRetryOptions(attempts=1)))

    def use_fallback(self) -> bool:
        """Switch to the fallback model (once). Returns False if there is none left."""
        if not self.fallback_model:
            return False
        self.model, self.fallback_model = self.fallback_model, ""
        return True

    def generate(self, system: str, prompt: str, schema) -> AIResponse:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=AI_TEMPERATURE,
            response_mime_type="application/json", response_schema=schema,
            # No tools are used; switch the SDK's automatic function calling off explicitly.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt,
                                                            config=config)
        except Exception as exc:  # noqa: BLE001
            raise classify_error(exc) from exc
        latency = int((time.perf_counter() - start) * 1000)
        usage = getattr(response, "usage_metadata", None)
        return AIResponse(text=response.text or "", model=self.model,
                          input_tokens=getattr(usage, "prompt_token_count", None),
                          output_tokens=getattr(usage, "candidates_token_count", None),
                          latency_ms=latency)


def client_from_settings(settings) -> GeminiClient:
    """The real client, or a typed error explaining what is missing."""
    if not settings.ai_enabled:
        raise ai_error("disabled")
    return GeminiClient(settings.gemini_api_key, settings.gemini_model,
                        fallback_model=getattr(settings, "gemini_fallback_model", ""))


def list_generation_models(api_key: str) -> list[str]:
    """Names of models this key may use for text generation (a diagnostic request; it does not
    generate any content). Used only when the owner asks, e.g. after a 'model not found' error."""
    if not api_key:
        raise ai_error("missing_key")
    from google import genai
    client = genai.Client(api_key=api_key)
    try:
        names = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(str(m.name).removeprefix("models/"))
        return sorted(names)
    except Exception as exc:  # noqa: BLE001
        raise classify_error(exc) from exc
