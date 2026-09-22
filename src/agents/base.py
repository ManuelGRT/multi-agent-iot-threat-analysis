# src/agents/base.py
from __future__ import annotations

import asyncio
import json
import httpx
import os
import re
from typing import Any

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - exercised only when optional deps are absent
    HumanMessage = None
    SystemMessage = None
    ChatOllama = None


class OllamaChatAgent:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float | None = None,
        backend: str | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        self.backend = (backend or os.getenv("LLM_BACKEND", "langchain")).strip().lower()
        self.reasoning = os.getenv("OLLAMA_REASONING", "false").strip().lower() in {"1", "true", "yes"}

    async def invoke_json(self, system_prompt: str, user_payload: dict[str, Any], json_schema: dict[str, Any]) -> dict[str, Any]:
        if self.backend != "httpx" and ChatOllama is not None:
            return await self._invoke_json_langchain(system_prompt, user_payload, json_schema)
        return await self._invoke_json_httpx(system_prompt, user_payload, json_schema)

    async def _invoke_json_langchain(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        model = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0,
            reasoning=self.reasoning,
            num_ctx=8192,
            async_client_kwargs={"timeout": self.timeout_seconds},
        )
        structured_model = model.with_structured_output(json_schema, method="json_schema")
        response = await structured_model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=True, default=str)),
        ])
        if not isinstance(response, dict):
            raise ValueError(f"LangChain Ollama response was not a JSON object: {response}")
        return response

    async def _invoke_json_httpx(self, system_prompt: str, user_payload: dict[str, Any], json_schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, default=str)},
            ],
            "format": json_schema,
            "stream": False,
            "think": self.reasoning,
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
            content = data["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ollama response was not valid JSON: {content}") from exc


class OpenAICompatibleChatAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | tuple[str, ...],
        provider_name: str,
        timeout_env: str | tuple[str, ...] | None = None,
        max_tokens_env: str | tuple[str, ...] | None = None,
        default_timeout_seconds: float = 240,
        default_max_tokens: int = 2200,
        timeout_seconds: float | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        json_mode_env: str | tuple[str, ...] | None = None,
        rate_limit_retries_env: str | tuple[str, ...] | None = None,
        connection_retries_env: str | tuple[str, ...] | None = None,
        retry_status_codes_env: str | tuple[str, ...] | None = None,
        retry_wait_seconds_env: str | tuple[str, ...] | None = None,
        reasoning_effort_env: str | tuple[str, ...] | None = None,
        include_reasoning_env: str | tuple[str, ...] | None = None,
        include_schema_in_prompt_env: str | tuple[str, ...] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self.api_key_env = _as_env_tuple(api_key_env)
        self.timeout_env = _as_env_tuple(timeout_env)
        self.max_tokens_env = _as_env_tuple(max_tokens_env)
        self.json_mode_env = _as_env_tuple(json_mode_env)
        self.rate_limit_retries_env = _as_env_tuple(rate_limit_retries_env)
        self.connection_retries_env = _as_env_tuple(connection_retries_env)
        self.retry_status_codes_env = _as_env_tuple(retry_status_codes_env)
        self.retry_wait_seconds_env = _as_env_tuple(retry_wait_seconds_env)
        self.reasoning_effort_env = _as_env_tuple(reasoning_effort_env)
        self.include_reasoning_env = _as_env_tuple(include_reasoning_env)
        self.include_schema_in_prompt_env = _as_env_tuple(include_schema_in_prompt_env)
        self.timeout_seconds = timeout_seconds or float(
            _first_env_value(self.timeout_env) or default_timeout_seconds
        )
        self.max_tokens = int(_first_env_value(self.max_tokens_env) or default_max_tokens)
        self.api_key = api_key or _first_env_value(self.api_key_env)
        self.extra_headers = extra_headers or {}
        self.json_mode = _env_bool(self.json_mode_env, default=True)
        # Los proveedores remotos aplican limites de tasa y pueden sufrir
        # fallos transitorios de red. Ninguno debe degradar silenciosamente el
        # agente a su ruta de reserva sin agotar antes sus propios reintentos.
        self.rate_limit_retries = int(_first_env_value(self.rate_limit_retries_env) or 3)
        self.connection_retries = max(
            0,
            int(_first_env_value(self.connection_retries_env) or 3),
        )
        self.retry_status_codes = _retry_status_codes(
            _first_env_value(self.retry_status_codes_env),
            default={429},
        )
        self.retry_wait_seconds = float(_first_env_value(self.retry_wait_seconds_env) or 0)

    async def invoke_json(self, system_prompt: str, user_payload: dict[str, Any], json_schema: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            env_hint = " or ".join(self.api_key_env)
            raise ValueError(f"{env_hint} is required for {self.provider_name}")

        include_schema_in_prompt = _env_bool(self.include_schema_in_prompt_env, default=True)
        user_message = {
            "input": user_payload,
            "instruction": (
                "Return only one valid JSON object matching json_schema."
                if include_schema_in_prompt
                else "Return only one valid JSON object for the canonical event contract described by the system prompt."
            ),
        }
        if include_schema_in_prompt:
            user_message["json_schema"] = json_schema
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_message, ensure_ascii=True, default=str),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        reasoning_effort = _first_env_value(self.reasoning_effort_env)
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort.strip()
        include_reasoning = _first_env_value(self.include_reasoning_env)
        if include_reasoning is not None:
            body["include_reasoning"] = include_reasoning.strip().lower() in {"1", "true", "yes", "on"}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            connection_retries = 0
            status_retries = 0
            while True:
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=body,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if connection_retries >= self.connection_retries:
                        raise
                    await asyncio.sleep(
                        _connection_retry_wait_seconds(
                            connection_retries,
                            self.retry_wait_seconds,
                        )
                    )
                    connection_retries += 1
                    continue
                if (
                    getattr(response, "status_code", None) in self.retry_status_codes
                    and status_retries < self.rate_limit_retries
                ):
                    status_retries += 1
                    await asyncio.sleep(_retry_wait_seconds(response, self.retry_wait_seconds))
                    continue
                break
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _compact_http_error_detail(response)
                raise httpx.HTTPStatusError(
                    f"{exc} | {detail}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            data = response.json()
            content = data["choices"][0]["message"].get("content", "")
            parsed = self._extract_json_object(self._message_content_to_text(content))
            if parsed is None:
                raise ValueError(f"{self.provider_name} response was not valid JSON: {content}")
            return parsed

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content, ensure_ascii=True, default=str)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for index, char in enumerate(cleaned[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start : index + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None


class OpenRouterChatAgent(OpenAICompatibleChatAgent):
    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env="OPENROUTER_API_KEY",
            provider_name="OpenRouterChatAgent",
            timeout_env="OPENROUTER_TIMEOUT_SECONDS",
            max_tokens_env=("OPENROUTER_MAX_TOKENS", "OPENAI_COMPATIBLE_MAX_TOKENS"),
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            json_mode_env=("OPENROUTER_JSON_MODE", "OPENAI_COMPATIBLE_JSON_MODE"),
            rate_limit_retries_env=("OPENROUTER_RATE_LIMIT_RETRIES", "OPENAI_COMPATIBLE_RATE_LIMIT_RETRIES"),
            connection_retries_env=("OPENROUTER_CONNECTION_RETRIES", "OPENAI_COMPATIBLE_CONNECTION_RETRIES"),
            retry_status_codes_env=("OPENROUTER_RETRY_STATUS_CODES", "OPENAI_COMPATIBLE_RETRY_STATUS_CODES"),
            retry_wait_seconds_env=("OPENROUTER_RETRY_WAIT_SECONDS", "OPENAI_COMPATIBLE_RETRY_WAIT_SECONDS"),
            extra_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost/tfm_multiagent"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "tfm_multiagent"),
            },
        )


class GroqChatAgent(OpenAICompatibleChatAgent):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env="GROQ_API_KEY",
            provider_name="GroqChatAgent",
            timeout_env="GROQ_TIMEOUT_SECONDS",
            max_tokens_env=("GROQ_MAX_TOKENS", "OPENAI_COMPATIBLE_MAX_TOKENS"),
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            json_mode_env=("GROQ_JSON_MODE", "OPENAI_COMPATIBLE_JSON_MODE"),
            rate_limit_retries_env=("GROQ_RATE_LIMIT_RETRIES", "OPENAI_COMPATIBLE_RATE_LIMIT_RETRIES"),
            connection_retries_env=("GROQ_CONNECTION_RETRIES", "OPENAI_COMPATIBLE_CONNECTION_RETRIES"),
            retry_status_codes_env=("GROQ_RETRY_STATUS_CODES", "OPENAI_COMPATIBLE_RETRY_STATUS_CODES"),
            retry_wait_seconds_env=("GROQ_RETRY_WAIT_SECONDS", "OPENAI_COMPATIBLE_RETRY_WAIT_SECONDS"),
            reasoning_effort_env=("GROQ_REASONING_EFFORT", "OPENAI_COMPATIBLE_REASONING_EFFORT"),
            include_reasoning_env=("GROQ_INCLUDE_REASONING", "OPENAI_COMPATIBLE_INCLUDE_REASONING"),
            include_schema_in_prompt_env=("GROQ_INCLUDE_SCHEMA_IN_PROMPT", "OPENAI_COMPATIBLE_INCLUDE_SCHEMA_IN_PROMPT"),
        )


class MistralChatAgent(OpenAICompatibleChatAgent):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.mistral.ai/v1",
        timeout_seconds: float | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env="MISTRAL_API_KEY",
            provider_name="MistralChatAgent",
            timeout_env="MISTRAL_TIMEOUT_SECONDS",
            max_tokens_env=("MISTRAL_MAX_TOKENS", "OPENAI_COMPATIBLE_MAX_TOKENS"),
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            json_mode_env=("MISTRAL_JSON_MODE", "OPENAI_COMPATIBLE_JSON_MODE"),
            rate_limit_retries_env=("MISTRAL_RATE_LIMIT_RETRIES", "OPENAI_COMPATIBLE_RATE_LIMIT_RETRIES"),
            connection_retries_env=("MISTRAL_CONNECTION_RETRIES", "OPENAI_COMPATIBLE_CONNECTION_RETRIES"),
            retry_status_codes_env=("MISTRAL_RETRY_STATUS_CODES", "OPENAI_COMPATIBLE_RETRY_STATUS_CODES"),
            retry_wait_seconds_env=("MISTRAL_RETRY_WAIT_SECONDS", "OPENAI_COMPATIBLE_RETRY_WAIT_SECONDS"),
        )


class GoogleAIStudioChatAgent(OpenAICompatibleChatAgent):
    def __init__(
        self,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds: float | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            provider_name="GoogleAIStudioChatAgent",
            timeout_env=("GOOGLE_AI_STUDIO_TIMEOUT_SECONDS", "GEMINI_TIMEOUT_SECONDS", "GOOGLE_TIMEOUT_SECONDS"),
            max_tokens_env=(
                "GOOGLE_AI_STUDIO_MAX_TOKENS",
                "GEMINI_MAX_TOKENS",
                "GOOGLE_MAX_TOKENS",
                "OPENAI_COMPATIBLE_MAX_TOKENS",
            ),
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            json_mode_env=("GOOGLE_AI_STUDIO_JSON_MODE", "GEMINI_JSON_MODE", "GOOGLE_JSON_MODE", "OPENAI_COMPATIBLE_JSON_MODE"),
            rate_limit_retries_env=(
                "GOOGLE_AI_STUDIO_RATE_LIMIT_RETRIES",
                "GEMINI_RATE_LIMIT_RETRIES",
                "GOOGLE_RATE_LIMIT_RETRIES",
                "OPENAI_COMPATIBLE_RATE_LIMIT_RETRIES",
            ),
            connection_retries_env=(
                "GOOGLE_AI_STUDIO_CONNECTION_RETRIES",
                "GEMINI_CONNECTION_RETRIES",
                "GOOGLE_CONNECTION_RETRIES",
                "OPENAI_COMPATIBLE_CONNECTION_RETRIES",
            ),
            retry_status_codes_env=(
                "GOOGLE_AI_STUDIO_RETRY_STATUS_CODES",
                "GEMINI_RETRY_STATUS_CODES",
                "GOOGLE_RETRY_STATUS_CODES",
                "OPENAI_COMPATIBLE_RETRY_STATUS_CODES",
            ),
            retry_wait_seconds_env=(
                "GOOGLE_AI_STUDIO_RETRY_WAIT_SECONDS",
                "GEMINI_RETRY_WAIT_SECONDS",
                "GOOGLE_RETRY_WAIT_SECONDS",
                "OPENAI_COMPATIBLE_RETRY_WAIT_SECONDS",
            ),
        )


class TransformersChatAgent:
    def __init__(
        self,
        model: str,
        adapter_path: str | None = None,
        timeout_seconds: float | None = None,
    ):
        del timeout_seconds
        self.model = model
        self.adapter_path = adapter_path
        self.max_new_tokens = int(os.getenv("TRANSFORMERS_MAX_NEW_TOKENS", "1800"))
        self.device = os.getenv("TRANSFORMERS_DEVICE", "auto").strip().lower()
        self.torch_dtype = os.getenv("TRANSFORMERS_DTYPE", "auto").strip().lower()
        self._loaded = None

    async def invoke_json(self, system_prompt: str, user_payload: dict[str, Any], json_schema: dict[str, Any]) -> dict[str, Any]:
        tokenizer, model = self._load()
        prompt = self._format_prompt(tokenizer, system_prompt, user_payload, json_schema)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        content = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _extract_json_object(content)

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional llm-training extra
            raise ImportError("Install the 'llm-training' extra to use TransformersChatAgent") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        dtype = _torch_dtype(torch, self.torch_dtype, use_cuda)
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if use_cuda:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - optional llm-training extra
                raise ImportError("Install peft to load LoRA adapters") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)
        if not use_cuda:
            model = model.to("cpu")
        model.eval()
        self._loaded = (tokenizer, model)
        return self._loaded

    def _format_prompt(
        self,
        tokenizer: Any,
        system_prompt: str,
        user_payload: dict[str, Any],
        json_schema: dict[str, Any],
    ) -> str:
        user_content = json.dumps(
            {
                "input": user_payload,
                "json_schema": json_schema,
                "instruction": "Return only one valid JSON object matching json_schema.",
            },
            ensure_ascii=True,
            default=str,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_content}\n\nASSISTANT:\n"


def _as_env_tuple(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _first_env_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_bool(names: tuple[str, ...], default: bool = False) -> bool:
    value = _first_env_value(names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _retry_status_codes(value: str | None, default: set[int]) -> set[int]:
    if not value:
        return set(default)
    codes = set()
    for item in re.split(r"[,;\s]+", value.strip()):
        if not item:
            continue
        try:
            codes.add(int(item))
        except ValueError:
            continue
    return codes or set(default)


def _compact_http_error_detail(response: httpx.Response) -> str:
    headers = {}
    for key in (
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ):
        value = response.headers.get(key)
        if value is not None:
            headers[key] = value
    text = response.text.strip()
    if len(text) > 1200:
        text = text[:1200] + "...[truncated]"
    return json.dumps(
        {
            "status_code": response.status_code,
            "headers": headers,
            "body": text,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _retry_wait_seconds(response: httpx.Response, configured_wait_seconds: float = 0) -> float:
    if configured_wait_seconds > 0:
        return configured_wait_seconds

    candidates = [
        _duration_to_seconds(response.headers.get("retry-after")),
        _duration_to_seconds(response.headers.get("x-ratelimit-reset-tokens")),
        _duration_to_seconds(response.headers.get("x-ratelimit-reset-requests")),
    ]
    try:
        body = response.json()
    except ValueError:
        body = {}
    message = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
    match = re.search(r"try again in\s+([0-9.]+)\s*s", message, flags=re.IGNORECASE)
    if match:
        candidates.append(float(match.group(1)))
    if response.status_code in {500, 502, 503, 504}:
        candidates.append(10.0)
    wait = max([value for value in candidates if value is not None] or [1.0])
    return min(max(wait + 0.75, 1.0), 120.0)


def _connection_retry_wait_seconds(
    retry_index: int,
    configured_wait_seconds: float = 0,
) -> float:
    """Espera breve y acotada para DNS/conexión, independiente de HTTP."""
    base_wait = configured_wait_seconds if configured_wait_seconds > 0 else 1.0
    return min(base_wait * (2 ** max(retry_index, 0)), 10.0)


def _duration_to_seconds(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"([0-9.]+)\s*(ms|s|m|h)", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        return amount / 1000
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    return amount * 3600


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Transformers response was not valid JSON: {content}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"Transformers response was not a JSON object: {content}")
    return data


def _torch_dtype(torch: Any, requested: str, use_cuda: bool):
    if requested == "float32" or not use_cuda:
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
