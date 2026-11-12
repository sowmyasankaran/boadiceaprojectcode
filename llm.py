import os
import json
import time
import random

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.0  # deterministic-leaning default; override per call

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0  # doubles each retry, plus jitter

_client = None


_ACTIVE_MODEL = None


def available() -> bool:

    return bool(os.environ.get("OPENROUTER_API_KEY"))


def provider() -> str:

    return "openrouter"


def default_model(model: str = None) -> str:
    return model or DEFAULT_MODEL


def set_active_model(model) -> None:
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model


def get_active_model():
    return _ACTIVE_MODEL


def reference_model() -> str:
    return os.environ.get("OPENROUTER_SIMULATOR_MODEL") or DEFAULT_MODEL


def _resolve_model(model: str = None) -> str:
    return model or _ACTIVE_MODEL or DEFAULT_MODEL


def _client_instance():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Get a key at "
                "https://openrouter.ai/keys and export it before running."
            )
        from openai import OpenAI
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "canrisk-conversational-agent"),
            },
        )
    return _client


def _is_transient(exc, openai_errors) -> bool:
    (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) = openai_errors
    return isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError))


def _create_chat_completion(**kwargs):

    from openai import (
        RateLimitError, APITimeoutError, APIConnectionError,
        InternalServerError, BadRequestError,
    )
    client = _client_instance()
    openai_errors = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

    last_exc = None
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as e:

            if "temperature" in kwargs and "temperature" in str(e).lower():
                kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
                continue
            raise
        except openai_errors as e:
            last_exc = e
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    raise last_exc



def call_text(system: str, messages: list, max_tokens: int = MAX_TOKENS,
              temperature: float = DEFAULT_TEMPERATURE, model: str = None,
              provider_override: str = None) -> str:
    mdl = _resolve_model(model)
    kwargs = dict(
        model=mdl, max_completion_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system}] + messages,
    )
    resp = _create_chat_completion(**kwargs)
    return resp.choices[0].message.content or ""

def call_with_tool(system: str, messages: list, tool: dict,
                   max_tokens: int = MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE,
                   model: str = None, provider_override: str = None) -> dict:

    mdl = _resolve_model(model)
    openai_tool = {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }
    kwargs = dict(
        model=mdl, max_completion_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system}] + messages,
        tools=[openai_tool],
        tool_choice={"type": "function", "function": {"name": tool["name"]}},
    )
    resp = _create_chat_completion(**kwargs)
    msg = resp.choices[0].message
    if msg.tool_calls:
        try:
            return json.loads(msg.tool_calls[0].function.arguments)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}