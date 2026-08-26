"""OpenAI-compatible LLM provider interface and caching layer."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

PROFILES: dict[str, tuple[str, str]] = {
    "ollama": ("http://localhost:11434/v1", ""),
    "replay": ("http://replay.invalid/v1", ""),
}

LLM_CACHE = Path(os.environ.get("LLM_CACHE", "fixtures/llm"))
STRUCTURED_MAX_TOKENS = int(os.environ.get("STRUCTURED_MAX_TOKENS", "8192"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
LLM_SCHEMA_RETRIES = int(os.environ.get("LLM_SCHEMA_RETRIES", "2"))

ROLE_CHAINS: dict[str, list[str]] = {
    "vision": ["gemma-4"],
}

SAMPLING: list[tuple[str, dict[str, float]]] = []

T = TypeVar("T", bound=BaseModel)

REPLAY_MANIFEST = LLM_CACHE / "models.json"


def replay_only() -> bool:
    return os.environ.get("LLM_PROFILE", "ollama") == "replay"


def sampling_for(model_id: str) -> dict[str, float]:
    lowered = model_id.lower()
    for needle, params in SAMPLING:
        if needle in lowered:
            return dict(params)
    return {}


@dataclass
class LLM:
    """Resolved client and model configuration for a specific role."""

    role: str
    model: str
    client: OpenAI
    params: dict[str, float] = field(default_factory=dict)

    def chat(
        self,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send chat completion request, checking disk cache first."""
        path = self._cache_path(system, user, images)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["content"]

        if replay_only():
            raise FileNotFoundError(
                f"LLM_PROFILE=replay but no fixture for role {self.role!r} at {path}."
            )

        content: list[dict[str, Any]] | str
        if images:
            content = [{"type": "text", "text": user}]
            for raw in images:
                b64 = base64.b64encode(raw).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
        else:
            content = user

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        answer = ""
        finish_reason: str | None = None
        final_budget = max_tokens
        budgets = (max_tokens, max_tokens * 4)
        for index, budget in enumerate(budgets):
            final_budget = budget
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=budget,
                **self.params,
            )
            choice = resp.choices[0]
            answer = choice.message.content or ""
            finish_reason = choice.finish_reason
            if finish_reason != "length":
                break
            reasoning = getattr(choice.message, "reasoning", None) or ""
            if index + 1 < len(budgets):
                log.warning(
                    "[%s] %s truncated at max_tokens=%d "
                    "(%d content chars, %d reasoning chars); retrying at %d",
                    self.role, self.model, budget, len(answer), len(reasoning),
                    budgets[index + 1],
                )

        if finish_reason == "length":
            raise RuntimeError(
                f"{self.role}: {self.model} output was truncated at "
                f"max_tokens={final_budget}; incomplete content was not cached"
            )
        if not answer.strip():
            raise RuntimeError(
                f"{self.role}: {self.model} returned empty content "
                f"(finish_reason={finish_reason!r})"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        return answer

    def _cache_path(self, system: str, user: str, images: list[bytes] | None) -> Path:
        digest = hashlib.sha256()
        digest.update(self.role.encode())
        digest.update(system.encode())
        digest.update(user.encode())
        for raw in images or []:
            digest.update(hashlib.sha256(raw).digest())
        return LLM_CACHE / self.role / f"{digest.hexdigest()[:20]}.json"

    def _discard_invalid_live_cache(
        self, system: str, user: str, images: list[bytes] | None
    ) -> None:
        """Remove only malformed live output produced by the currently selected model."""
        if replay_only():
            return
        path = self._cache_path(system, user, images)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if payload.get("model") == self.model:
            path.unlink(missing_ok=True)
            log.warning("[%s] discarded invalid cache entry %s", self.role, path)

    def structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        images: list[bytes] | None = None,
        retries: int | None = None,
        max_tokens: int = STRUCTURED_MAX_TOKENS,
    ) -> T:
        """Query completion and validate against a Pydantic schema with retries."""
        instruction = (
            f"{user}\n\nRespond with a single JSON object matching this schema. "
            f"No prose, no markdown fences.\n{json.dumps(schema.model_json_schema())}"
        )
        last_error = ""
        schema_retries = LLM_SCHEMA_RETRIES if retries is None else retries
        for attempt in range(schema_retries + 1):
            prompt = instruction if not last_error else (
                f"{instruction}\n\nYour previous reply was rejected: {last_error}\n"
                "Return only valid JSON for the schema."
            )
            raw = self.chat(system, prompt, images=images, max_tokens=max_tokens)
            try:
                return schema.model_validate_json(_strip_fences(raw))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:400]
                self._discard_invalid_live_cache(system, prompt, images)
                log.warning(
                    "[%s] schema repair %d/%d: %s",
                    self.role, attempt + 1, schema_retries + 1, last_error,
                )
        raise ValueError(
            f"{self.role}: could not obtain valid {schema.__name__} after "
            f"{schema_retries + 1} attempts: {last_error}"
        )


def _strip_fences(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
    start, end = body.find("{"), body.rfind("}")
    return body[start : end + 1] if start != -1 and end != -1 else body


class Provider:
    """Manages LLM instances and model resolution for configured roles."""

    def __init__(self, profile: str | None = None):
        self.profile = profile or os.environ.get("LLM_PROFILE", "ollama")
        if self.profile not in PROFILES:
            raise ValueError(
                f"Unknown LLM_PROFILE {self.profile!r}. Options: {list(PROFILES)}"
            )
        base_url, key_var = PROFILES[self.profile]
        override = None
        if self.profile == "ollama":
            override = os.environ.get("OLLAMA_BASE_URL")
        self.base_url = os.environ.get("LLM_BASE_URL") or override or base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get(key_var) or "not-needed",
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
        self._available: set[str] | None = None
        self._cache: dict[str, LLM] = {}

    def available(self) -> set[str]:
        if self._available is not None:
            return self._available

        if self.profile == "replay":
            if REPLAY_MANIFEST.exists():
                recorded = json.loads(REPLAY_MANIFEST.read_text(encoding="utf-8"))
                self._available = {
                    entry["model"] for entry in recorded.get("roles", {}).values()
                }
            else:
                self._available = {f"replay-{role}" for role in ROLE_CHAINS}
            return self._available

        try:
            self._available = {m.id for m in self.client.models.list().data}
        except Exception as exc:
            raise RuntimeError(
                f"Could not list models at {self.base_url}. Original error: {exc}"
            ) from exc
        return self._available

    def get(self, role: str) -> LLM:
        if role != "vision":
            raise ValueError(f"Only the 'vision' role is configured, not {role!r}")
        if role in self._cache:
            return self._cache[role]

        override = os.environ.get("MODEL_VISION")
        model = override or self._resolve(role)
        llm = LLM(role=role, model=model, client=self.client, params=sampling_for(model))
        self._cache[role] = llm
        log.info("role=%s -> model=%s (%s)", role, model, self.profile)
        return llm

    def _resolve(self, role: str) -> str:
        chain = ROLE_CHAINS.get(role)
        if not chain:
            raise ValueError(f"No model chain for role {role!r}")
        served = sorted(self.available())

        if self.profile == "replay":
            if REPLAY_MANIFEST.exists():
                recorded = json.loads(REPLAY_MANIFEST.read_text(encoding="utf-8"))
                entry = recorded.get("roles", {}).get(role)
                if entry:
                    return entry["model"]
            return f"replay-{role}"
        for needle in chain:
            for model_id in served:
                lowered = model_id.lower()
                if needle in lowered:
                    return model_id
        raise RuntimeError(
            f"No model for role {role!r} at {self.base_url}.\n"
            f"  wanted: {chain}\n"
            f"  available: {served}"
        )

    def manifest(self) -> dict[str, Any]:
        """Generate provider execution manifest."""
        body = {
            "profile": self.profile,
            "base_url": self.base_url,
            "transport": {
                "timeout_seconds": LLM_TIMEOUT_SECONDS,
                "max_retries": LLM_MAX_RETRIES,
                "schema_retries": LLM_SCHEMA_RETRIES,
                "structured_max_tokens": STRUCTURED_MAX_TOKENS,
                "truncation_retry_multiplier": 4,
            },
            "roles": {
                llm.role: {"model": llm.model, "sampling": llm.params}
                for llm in self._cache.values()
            },
        }
        if self.profile != "replay" and body["roles"]:
            try:
                REPLAY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
                REPLAY_MANIFEST.write_text(json.dumps(body, indent=2), encoding="utf-8")
            except OSError as exc:
                log.warning("could not write %s: %s", REPLAY_MANIFEST, exc)
        return body


def jpeg_bytes(path: str, max_side: int = 896) -> bytes:
    """Downscale an image to JPEG bytes for vision model ingestion."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
