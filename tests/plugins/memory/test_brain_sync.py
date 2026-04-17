import json
import os

from plugins.memory.brain_sync import BrainSyncMemoryProvider


class FakeBrainClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        if name.endswith("read_note"):
            return {"content": [{"type": "text", "text": f"read:{arguments['filePath']}"}]}
        if name.endswith("search_notes"):
            return {"content": [{"type": "text", "text": f"search:{arguments['query']}"}]}
        return {"content": [{"type": "text", "text": json.dumps({"ok": True})}]}


class temp_env:
    def __init__(self, **values):
        self.values = values
        self.original = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_provider(source_model="gpt-5.4-mini", source_provider="openai"):
    provider = BrainSyncMemoryProvider()
    provider.initialize(
        session_id="session-1",
        source_runtime="hermes",
        source_model=source_model,
        source_provider=source_provider,
    )
    provider._client = FakeBrainClient()
    return provider


def test_brain_sync_provider_requires_brain_env():
    with temp_env(BRAIN_URL=None, BRAIN_TOKEN=None):
        provider = BrainSyncMemoryProvider()
        assert provider.is_available() is False

    with temp_env(BRAIN_URL="https://brain.example.com/mcp", BRAIN_TOKEN=None):
        provider = BrainSyncMemoryProvider()
        assert provider.is_available() is True


def test_system_prompt_and_prefetch_read_from_brain():
    with temp_env(BRAIN_URL="https://brain.example.com/mcp", BRAIN_TOKEN="brain-token"):
        provider = build_provider()

        prompt_block = provider.system_prompt_block()
        assert "Shared Brain notes" in prompt_block
        assert "FOR-AI-AGENTS.md" in prompt_block

        recall = provider.prefetch("memory policy")
        assert "memory policy" in recall
        assert provider._client.calls[-1][0] == "brain_search_notes"


def test_untrusted_memory_write_creates_brain_proposal():
    with temp_env(
        BRAIN_URL="https://brain.example.com/mcp",
        BRAIN_TOKEN="brain-token",
        BRAIN_MEMORY_TRUSTED_MODELS="gpt-5.4",
    ):
        provider = build_provider(source_model="gpt-5.4-mini", source_provider="openai")
        provider.on_memory_write(
            "add",
            "memory",
            "Hermes cheap routes should propose Brain updates.",
            source_runtime="hermes",
            source_provider="openai",
            source_model="gpt-5.4-mini",
            session_id="session-123",
        )

        name, args = provider._client.calls[-1]
        assert name == "brain_queue_memory_proposal"
        assert args["sourceModel"] == "gpt-5.4-mini"
        assert args["sourceProvider"] == "openai"
        assert args["sourceRuntime"] == "hermes"
        assert args["targetPath"] == "Global/AI-Learnings.md"


def test_trusted_memory_write_appends_final_memory_directly():
    with temp_env(
        BRAIN_URL="https://brain.example.com/mcp",
        BRAIN_TOKEN="brain-token",
        BRAIN_MEMORY_TRUSTED_MODELS="gpt-5.4",
    ):
        provider = build_provider(source_model="gpt-5.4", source_provider="openai")
        provider.on_memory_write(
            "add",
            "user",
            "Vincent prefers proposal audit trails in Brain.",
            source_runtime="hermes",
            source_provider="openai",
            source_model="gpt-5.4",
        )

        name, args = provider._client.calls[-1]
        assert name == "brain_append_note"
        assert args["filePath"] == "MEMORY.md"
        assert "proposal audit trails" in args["content"]
