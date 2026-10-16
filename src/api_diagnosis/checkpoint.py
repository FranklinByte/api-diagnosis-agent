"""Checkpoint and human-approval primitives for the V2 Agent boundary."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - optional during minimal V0 installs
    InMemorySaver = None  # type: ignore[assignment,misc]


def new_memory_saver() -> Any:
    """Create the LangGraph in-memory checkpointer used by tests and demos."""

    if InMemorySaver is None:
        raise RuntimeError("langgraph is required for Agent checkpoints")
    return InMemorySaver()


@dataclass
class ApprovalGate:
    """Explicit HITL gate for replay or other non-idempotent actions."""

    _pending: dict[str, dict[str, Any]] = field(default_factory=dict)

    def request(self, action: str, *, details: Optional[dict[str, Any]] = None) -> str:
        token = secrets.token_urlsafe(18)
        self._pending[token] = {
            "action": action,
            "details": details or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved": False,
        }
        return token

    def approve(self, token: str) -> bool:
        item = self._pending.get(token)
        if item is None:
            return False
        item["approved"] = True
        item["approved_at"] = datetime.now(timezone.utc).isoformat()
        return True

    def reject(self, token: str) -> bool:
        return self._pending.pop(token, None) is not None

    def is_approved(self, token: str) -> bool:
        return bool(self._pending.get(token, {}).get("approved"))

    def describe(self, token: str) -> Optional[dict[str, Any]]:
        item = self._pending.get(token)
        return dict(item) if item else None


__all__ = ["ApprovalGate", "new_memory_saver"]
