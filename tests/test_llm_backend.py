import json

import pytest

from src.agents import base
from src.agents.base import GoogleAIStudioChatAgent, GroqChatAgent, MistralChatAgent, OllamaChatAgent


class FakeStructuredModel:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return {"event_id": "from-langchain"}


class FakeChatOllama:
    structured = FakeStructuredModel()

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_structured_output(self, schema, method="json_schema"):
        self.schema = schema
        self.method = method
        return self.structured


@pytest.mark.asyncio
async def test_ollama_chat_agent_uses_langchain_backend(monkeypatch):
    monkeypatch.setattr(base, "ChatOllama", FakeChatOllama)

    agent = OllamaChatAgent(model="test-model", backend="langchain")
    result = await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert result == {"event_id": "from-langchain"}


@pytest.mark.asyncio
async def test_ollama_chat_agent_can_force_httpx_backend(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": json.dumps({"event_id": "from-httpx"})}}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            self.url = url
            self.body = json
            return FakeResponse()

    monkeypatch.setattr(base.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(base, "ChatOllama", FakeChatOllama)

    agent = OllamaChatAgent(model="test-model", backend="httpx")
    result = await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert result == {"event_id": "from-httpx"}


@pytest.mark.asyncio
async def test_openai_compatible_agent_posts_chat_completion(monkeypatch):
    recorded = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": [
                            {"type": "text", "text": "```json\n{\"event_id\":\"from-api\"}\n```"}
                        ]
                    }
                }]
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            recorded["url"] = url
            recorded["headers"] = headers
            recorded["body"] = json
            return FakeResponse()

    monkeypatch.setattr(base.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")

    agent = GroqChatAgent(model="llama-3.1-8b-instant", timeout_seconds=7)
    result = await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert result == {"event_id": "from-api"}
    assert recorded["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert recorded["headers"]["Authorization"] == "Bearer test-secret"
    assert recorded["body"]["model"] == "llama-3.1-8b-instant"
    assert recorded["body"]["temperature"] == 0
    assert recorded["body"]["response_format"] == {"type": "json_object"}
    assert "json_schema" in recorded["body"]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_openai_compatible_agent_retries_rate_limit(monkeypatch):
    calls = []
    sleeps = []

    class FakeResponse:
        def __init__(self, status_code, payload, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            calls.append((url, headers, json))
            if len(calls) == 1:
                return FakeResponse(
                    429,
                    {"error": {"message": "Rate limit reached. Please try again in 2s."}},
                    {"retry-after": "2"},
                )
            return FakeResponse(
                200,
                {"choices": [{"message": {"content": "{\"event_id\":\"after-retry\"}"}}]},
            )

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(base.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setenv("GROQ_RATE_LIMIT_RETRIES", "1")

    agent = GroqChatAgent(model="openai/gpt-oss-120b", timeout_seconds=7)
    result = await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert result == {"event_id": "after-retry"}
    assert len(calls) == 2
    assert sleeps and sleeps[0] >= 2


@pytest.mark.parametrize(
    "error_type",
    [base.httpx.ConnectError, base.httpx.ConnectTimeout],
)
@pytest.mark.asyncio
async def test_openai_compatible_agent_retries_transient_connection_errors(
    monkeypatch,
    error_type,
):
    calls = []
    sleeps = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"event_id":"after-connect-retry"}'}}
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            calls.append((url, headers, json))
            if len(calls) <= 2:
                request = base.httpx.Request("POST", url)
                raise error_type("temporary connection failure", request=request)
            return FakeResponse()

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(base.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-secret")
    monkeypatch.setenv("MISTRAL_CONNECTION_RETRIES", "2")
    monkeypatch.delenv("MISTRAL_RETRY_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_RETRY_WAIT_SECONDS", raising=False)

    agent = MistralChatAgent(model="ministral-8b-latest", timeout_seconds=7)
    result = await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert result == {"event_id": "after-connect-retry"}
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_openai_compatible_agent_raises_after_connection_retries(monkeypatch):
    calls = []
    sleeps = []
    raised_errors = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            calls.append((url, headers, json))
            error = base.httpx.ConnectError(
                f"dns failure {len(calls)}",
                request=base.httpx.Request("POST", url),
            )
            raised_errors.append(error)
            raise error

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(base.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-secret")
    monkeypatch.setenv("MISTRAL_CONNECTION_RETRIES", "2")
    monkeypatch.delenv("MISTRAL_RETRY_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_RETRY_WAIT_SECONDS", raising=False)

    agent = MistralChatAgent(model="ministral-8b-latest", timeout_seconds=7)
    with pytest.raises(base.httpx.ConnectError, match="dns failure 3") as exc_info:
        await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert exc_info.value is raised_errors[-1]
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_connection_retries_do_not_retry_unconfigured_http_errors(monkeypatch):
    calls = []
    sleeps = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            calls.append((url, headers, json))
            return base.httpx.Response(
                401,
                request=base.httpx.Request("POST", url),
                json={"error": {"message": "invalid key"}},
            )

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(base.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-secret")
    monkeypatch.setenv("MISTRAL_CONNECTION_RETRIES", "3")

    agent = MistralChatAgent(model="ministral-8b-latest", timeout_seconds=7)
    with pytest.raises(base.httpx.HTTPStatusError):
        await agent.invoke_json("system", {"row": 1}, {"type": "object"})

    assert len(calls) == 1
    assert sleeps == []


def test_direct_api_agents_use_provider_env_keys(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.delenv("MISTRAL_CONNECTION_RETRIES", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_CONNECTION_RETRIES", raising=False)

    mistral = MistralChatAgent(model="mistral-small-latest")
    google = GoogleAIStudioChatAgent(model="gemini-2.5-flash")

    assert mistral.api_key == "mistral-secret"
    assert mistral.base_url == "https://api.mistral.ai/v1"
    assert mistral.connection_retries == 3
    assert google.api_key == "gemini-secret"
    assert google.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
