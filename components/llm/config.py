"""LLM clients by role, with request settings loaded from profiles.json.

The app talks to every LLM through the OpenAI-compatible chat API, so which model
answers and where it runs (local llama.cpp server, vLLM, a hosted API) are deployment
settings in .env, not code. Model-specific details (chat template, reasoning tags and
budget, sampling defaults like top_k) belong to the server; see README "LLM servers".

Per role, from .env:
    {ROLE}_API_BASE    server URL, e.g. http://127.0.0.1:8080/v1
    {ROLE}_MODEL_NAME  model name the server expects (llama-server: its --alias)
    {ROLE}_API_KEY     that server's API key; leave empty for local servers. Each role has
                       its own key so a key is only ever sent to the server it belongs to.
"""

import json
import os
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from llama_index.llms.openai_like import OpenAILike

PROFILES_PATH = Path(__file__).with_name("profiles.json")


class RoleSettings(BaseModel):
    """One role's request settings from profiles.json (roles: answer, hyde).

    temperature:    Sampling randomness (0 = always the most likely token). Low for HyDE.
                    The answer value is the answer model's recommended value
                    (phi-4-reasoning: 0.8); revisit it when the answer model changes.
    top_p:          Sample only from the most likely tokens whose probabilities sum to top_p.
                    None sends nothing, so the server default applies.
    max_tokens:     Max tokens generated per call. For a reasoning model this includes the
                    reasoning, which the server caps separately (--reasoning-budget), so it
                    must be the reasoning budget plus room for the answer.
    context_window: Total tokens per call (prompt + output). LlamaIndex packs retrieved chunks
                    into context_window - max_tokens, so it must not exceed the server's
                    --ctx-size. Nothing checks this automatically.
    timeout:        Seconds to wait for one call. CPU generation of max_tokens can take many
                    minutes, so the OpenAI client default (60 s) is far too short.
    reasoning_effort: OpenAI request parameter for hybrid models (thinking can be switched
                    off): "none" turns reasoning off, "low"/"medium"/"high" set its effort.
                    None sends nothing, so the model's default applies (some models think
                    by default, some do not). Reasoning tokens count toward max_tokens.
    """

    # extra="forbid" so a typo in profiles.json fails loudly instead of being ignored
    model_config = ConfigDict(extra="forbid")

    temperature: float
    top_p: float | None = None
    max_tokens: int
    context_window: int
    timeout: float = 1800.0
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None


def load_role(role: str) -> RoleSettings:
    """Load one role's settings from profiles.json."""
    roles = json.loads(PROFILES_PATH.read_text())["roles"]
    if role not in roles:
        raise KeyError(f"Unknown LLM role '{role}'. Available: {', '.join(roles)}")
    return RoleSettings.model_validate(roles[role])


def _role_env(role: str, name: str) -> str:
    key = f"{role.upper()}_{name}"
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"{key} is not set. See env.sample.")
    return value


def _role_api_key(role: str) -> str:
    # The OpenAI client requires a value; local servers ignore it
    return os.getenv(f"{role.upper()}_API_KEY") or "none"


def build_llm(role: str) -> OpenAILike:
    """Build the chat client for a role: settings from profiles.json, server from .env."""
    settings = load_role(role)
    # Optional request parameters: sent only when set, so the server default applies otherwise
    additional_kwargs = {}
    if settings.top_p is not None:
        additional_kwargs["top_p"] = settings.top_p
    if settings.reasoning_effort is not None:
        additional_kwargs["reasoning_effort"] = settings.reasoning_effort
    return OpenAILike(
        model=_role_env(role, "MODEL_NAME"),
        api_base=_role_env(role, "API_BASE"),
        api_key=_role_api_key(role),
        is_chat_model=True,
        context_window=settings.context_window,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        additional_kwargs=additional_kwargs,
        timeout=settings.timeout,
        # A retry re-runs the whole generation, which on CPU means many more minutes
        max_retries=0,
    )


def check_llm_server(role: str) -> None:
    """Fail early with a clear message if the role's server is not reachable.

    GET /models exists on every OpenAI-compatible server.
    """
    api_base = _role_env(role, "API_BASE")
    client = OpenAI(base_url=api_base, api_key=_role_api_key(role), timeout=5, max_retries=0)
    try:
        client.models.list()
    except Exception as exc:
        raise RuntimeError(
            f"LLM server for role '{role}' is not reachable at {api_base}. "
            "Start it first (README: LLM servers)."
        ) from exc
