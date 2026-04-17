"""Brain sync memory provider.

Uses AI Brain MCP as a shared durable memory backend so Hermes can read the same
notes and proposal queue that Codex-based assistants use.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_NOTE = "Global/Memory-Proposals.md"
DEFAULT_MEMORY_NOTE = "Global/AI-Learnings.md"
DEFAULT_USER_NOTE = "MEMORY.md"
DEFAULT_STARTUP_NOTES = ["FOR-AI-AGENTS.md", "MEMORY.md", "Global/AI-Learnings.md"]

READ_SCHEMAS = [
    {
        "name": "brain_search_notes",
        "description": "Search Brain markdown notes for relevant context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "dirPath": {"type": "string", "description": "Optional Brain-relative directory."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_read_note",
        "description": "Read a Brain markdown note by path.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "Brain-relative markdown path."},
            },
            "required": ["filePath"],
        },
    },
    {
        "name": "brain_list_memory_proposals",
        "description": "List memory proposals in the Brain queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Optional proposal status filter."},
                "targetPath": {"type": "string", "description": "Optional exact Brain note target path filter."},
            },
            "required": [],
        },
    },
    {
        "name": "brain_queue_memory_proposal",
        "description": "Create a pending memory proposal in the shared Brain queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "targetPath": {"type": "string", "description": "Brain-relative destination note path."},
                "targetOperation": {"type": "string", "description": "append or overwrite."},
                "proposedContent": {"type": "string", "description": "The proposed memory content."},
                "proposalNote": {"type": "string", "description": "Optional reviewer-facing note."},
            },
            "required": ["targetPath", "proposedContent"],
        },
    },
]

TRUSTED_ONLY_SCHEMAS = [
    {
        "name": "brain_write_note",
        "description": "Write a Brain markdown note directly.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["filePath", "content"],
        },
    },
    {
        "name": "brain_append_note",
        "description": "Append text to a Brain markdown note.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filePath", "content"],
        },
    },
    {
        "name": "brain_review_memory_proposal",
        "description": "Review a pending Brain memory proposal.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposalId": {"type": "string"},
                "decision": {"type": "string", "enum": ["approve", "approve-with-edit", "reject"]},
                "reviewerNote": {"type": "string"},
                "finalContent": {"type": "string"},
            },
            "required": ["proposalId", "decision"],
        },
    },
    {
        "name": "brain_merge_memory_proposal",
        "description": "Merge an approved Brain memory proposal into its destination note.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposalId": {"type": "string"},
                "targetPath": {"type": "string"},
                "targetOperation": {"type": "string", "enum": ["append", "overwrite"]},
                "finalContent": {"type": "string"},
                "mergeNote": {"type": "string"},
            },
            "required": ["proposalId"],
        },
    },
]


def _parse_list(value: str, fallback: Optional[List[str]] = None) -> List[str]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if items:
        return items
    return list(fallback or [])


class _BrainClient:
    def __init__(self, url: str, token: str = "", client_name: str = "hermes-brain-sync"):
        self._url = url
        self._token = token
        self._client_name = client_name
        self._session_id = ""
        self._request_counter = 0

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _parse_payload(raw_text: str, content_type: str = "") -> Dict[str, Any]:
        body = str(raw_text or "").strip()
        if not body:
            return {}
        if "text/event-stream" in str(content_type or "").lower():
            for line in body.splitlines():
                if line.strip().lower().startswith("data:"):
                    return json.loads(line.split(":", 1)[1].strip())
            raise RuntimeError("Brain returned an empty event-stream payload")
        return json.loads(body)

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = self._parse_payload(
                    response.read().decode("utf-8"),
                    response.headers.get("Content-Type", ""),
                )
                next_session = response.headers.get("mcp-session-id")
                if next_session:
                    self._session_id = next_session
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8")) if exc.fp else {}
            message = payload.get("error", {}).get("message") or str(exc)
            raise RuntimeError(message) from exc

        if payload.get("error"):
            raise RuntimeError(payload["error"].get("message") or "Brain request failed")

        return payload.get("result", {})

    def ensure_initialized(self) -> None:
        if self._session_id:
            return
        self._request_counter += 1
        self._post(
            {
                "jsonrpc": "2.0",
                "id": f"init-{self._request_counter}",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": self._client_name, "version": "1.0.0"},
                },
            }
        )

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.ensure_initialized()
        self._request_counter += 1
        return self._post(
            {
                "jsonrpc": "2.0",
                "id": f"tool-{self._request_counter}",
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments or {},
                },
            }
        )


class BrainSyncMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "brain_sync"

    def __init__(self):
        self._brain_url = ""
        self._brain_token = ""
        self._tool_prefix = "brain"
        self._queue_note_path = DEFAULT_QUEUE_NOTE
        self._target_memory_note = DEFAULT_MEMORY_NOTE
        self._target_user_note = DEFAULT_USER_NOTE
        self._proposal_mode = "manual_review"
        self._trusted_models: List[str] = []
        self._trusted_providers: List[str] = []
        self._startup_notes: List[str] = []
        self._client: Optional[_BrainClient] = None
        self._source_runtime = "hermes"
        self._source_provider = ""
        self._source_model = ""
        self._trusted_session = False

    def is_available(self) -> bool:
        return bool(os.getenv("BRAIN_URL", "").strip())

    def initialize(self, session_id: str, **kwargs) -> None:
        del session_id
        self._brain_url = os.getenv("BRAIN_URL", "").strip()
        self._brain_token = os.getenv("BRAIN_TOKEN", "").strip()
        self._tool_prefix = os.getenv("BRAIN_TOOL_PREFIX", "brain").strip()
        self._queue_note_path = os.getenv("BRAIN_MEMORY_QUEUE_NOTE", DEFAULT_QUEUE_NOTE).strip() or DEFAULT_QUEUE_NOTE
        self._target_memory_note = os.getenv("BRAIN_SYNC_TARGET_MEMORY_NOTE", DEFAULT_MEMORY_NOTE).strip() or DEFAULT_MEMORY_NOTE
        self._target_user_note = os.getenv("BRAIN_SYNC_TARGET_USER_NOTE", DEFAULT_USER_NOTE).strip() or DEFAULT_USER_NOTE
        self._proposal_mode = "auto_approve" if os.getenv("BRAIN_MEMORY_PROPOSAL_MODE", "").strip().lower() == "auto_approve" else "manual_review"
        self._trusted_models = _parse_list(os.getenv("BRAIN_MEMORY_TRUSTED_MODELS", "gpt-5.4"), ["gpt-5.4"])
        self._trusted_providers = [item.lower() for item in _parse_list(os.getenv("BRAIN_MEMORY_TRUSTED_PROVIDERS", ""), [])]
        self._startup_notes = _parse_list(os.getenv("BRAIN_SYNC_STARTUP_NOTES", ""), DEFAULT_STARTUP_NOTES)
        self._source_runtime = str(kwargs.get("source_runtime") or "hermes").strip() or "hermes"
        self._source_provider = str(kwargs.get("source_provider") or "").strip()
        self._source_model = str(kwargs.get("source_model") or "").strip()
        self._trusted_session = self._is_trusted(self._source_model, self._source_provider)
        self._client = _BrainClient(self._brain_url, self._brain_token)

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""

        chunks: List[str] = []
        for note_path in self._startup_notes:
            try:
                result = self._client.call_tool(self._tool_name("read_note"), {"filePath": note_path})
                text = self._extract_text(result)
                if text.strip():
                    chunks.append(f"## {note_path}\n{text.strip()}")
            except Exception:
                continue
        if not chunks:
            return ""
        return "Shared Brain notes:\n\n" + "\n\n".join(chunks)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        del session_id
        if not self._client or not query.strip():
            return ""
        try:
            result = self._client.call_tool(self._tool_name("search_notes"), {"query": query})
        except Exception:
            return ""
        text = self._extract_text(result).strip()
        if not text:
            return ""
        return f"Brain search context for '{query}':\n{text}"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        del user_content, assistant_content, session_id

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = list(READ_SCHEMAS)
        if self._trusted_session:
            schemas.extend(TRUSTED_ONLY_SCHEMAS)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        del kwargs
        if not self._client:
            return tool_error("Brain sync provider is not initialized")

        try:
            if tool_name == "brain_search_notes":
                result = self._client.call_tool(self._tool_name("search_notes"), {
                    "query": args.get("query", ""),
                    "dirPath": args.get("dirPath", "/") or "/",
                })
            elif tool_name == "brain_read_note":
                result = self._client.call_tool(self._tool_name("read_note"), {"filePath": args.get("filePath", "")})
            elif tool_name == "brain_list_memory_proposals":
                result = self._client.call_tool(
                    self._tool_name("list_memory_proposals"),
                    {
                        "queueNotePath": self._queue_note_path,
                        "status": args.get("status", ""),
                        "targetPath": args.get("targetPath", ""),
                    },
                )
            elif tool_name == "brain_queue_memory_proposal":
                result = self._client.call_tool(
                    self._tool_name("queue_memory_proposal"),
                    {
                        "queueNotePath": self._queue_note_path,
                        "targetPath": args.get("targetPath", ""),
                        "targetOperation": args.get("targetOperation", "append"),
                        "proposedContent": args.get("proposedContent", ""),
                        "proposalNote": args.get("proposalNote", ""),
                        "sourceRuntime": self._source_runtime,
                        "sourceProvider": self._source_provider,
                        "sourceModel": self._source_model,
                    },
                )
            elif tool_name == "brain_write_note":
                self._assert_trusted_tool(tool_name)
                result = self._client.call_tool(
                    self._tool_name("write_note"),
                    {
                        "filePath": args.get("filePath", ""),
                        "content": args.get("content", ""),
                        "overwrite": bool(args.get("overwrite", False)),
                    },
                )
            elif tool_name == "brain_append_note":
                self._assert_trusted_tool(tool_name)
                result = self._client.call_tool(
                    self._tool_name("append_note"),
                    {
                        "filePath": args.get("filePath", ""),
                        "content": args.get("content", ""),
                    },
                )
            elif tool_name == "brain_review_memory_proposal":
                self._assert_trusted_tool(tool_name)
                result = self._client.call_tool(
                    self._tool_name("review_memory_proposal"),
                    {
                        "queueNotePath": self._queue_note_path,
                        "proposalId": args.get("proposalId", ""),
                        "decision": args.get("decision", "approve"),
                        "reviewerNote": args.get("reviewerNote", ""),
                        "finalContent": args.get("finalContent", ""),
                        "reviewerRuntime": self._source_runtime,
                        "reviewerProvider": self._source_provider,
                        "reviewerModel": self._source_model,
                    },
                )
            elif tool_name == "brain_merge_memory_proposal":
                self._assert_trusted_tool(tool_name)
                result = self._client.call_tool(
                    self._tool_name("merge_memory_proposal"),
                    {
                        "queueNotePath": self._queue_note_path,
                        "proposalId": args.get("proposalId", ""),
                        "targetPath": args.get("targetPath", ""),
                        "targetOperation": args.get("targetOperation", "append"),
                        "finalContent": args.get("finalContent", ""),
                        "mergeNote": args.get("mergeNote", ""),
                    },
                )
            else:
                return tool_error(f"Unsupported Brain sync tool: {tool_name}")
        except Exception as exc:
            return tool_error(str(exc))

        return json.dumps(result, ensure_ascii=False)

    def on_memory_write(self, action: str, target: str, content: str, **kwargs) -> None:
        if action not in ("add", "replace") or not (content or "").strip() or not self._client:
            return

        source_runtime = str(kwargs.get("source_runtime") or self._source_runtime or "hermes").strip() or "hermes"
        source_provider = str(kwargs.get("source_provider") or self._source_provider or "").strip()
        source_model = str(kwargs.get("source_model") or self._source_model or "").strip()
        session_id = str(kwargs.get("session_id") or "").strip()
        target_path = self._target_user_note if target == "user" else self._target_memory_note
        trusted = self._is_trusted(source_model, source_provider)

        try:
            if trusted and self._proposal_mode == "auto_approve":
                self._client.call_tool(
                    self._tool_name("append_note"),
                    {"filePath": target_path, "content": f"{content.strip()}\n"},
                )
                return

            if trusted:
                self._client.call_tool(
                    self._tool_name("append_note"),
                    {"filePath": target_path, "content": f"{content.strip()}\n"},
                )
                return

            self._client.call_tool(
                self._tool_name("queue_memory_proposal"),
                {
                    "queueNotePath": self._queue_note_path,
                    "targetPath": target_path,
                    "targetOperation": "append",
                    "proposedContent": content.strip(),
                    "sourceRuntime": source_runtime,
                    "sourceProvider": source_provider,
                    "sourceModel": source_model,
                    "sourceSessionId": session_id,
                    "proposalNote": "Mirrored from Hermes built-in memory write.",
                },
            )
        except Exception:
            logger.debug("Brain sync memory mirror failed", exc_info=True)

    def _assert_trusted_tool(self, tool_name: str) -> None:
        if not self._trusted_session:
            raise RuntimeError(f"Tool requires a trusted Brain writer route: {tool_name}")

    def _tool_name(self, suffix: str) -> str:
        return f"{self._tool_prefix}_{suffix}" if self._tool_prefix else suffix

    def _is_trusted(self, model: str, provider: str) -> bool:
        normalized_provider = str(provider or "").strip().lower()
        normalized_model = str(model or "").strip()
        return normalized_model in self._trusted_models or normalized_provider in self._trusted_providers

    @staticmethod
    def _extract_text(result: Dict[str, Any]) -> str:
        content = result.get("content")
        if not isinstance(content, list):
            return json.dumps(result, ensure_ascii=False)
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)



def register(ctx) -> None:
    ctx.register_memory_provider(BrainSyncMemoryProvider())
