"""
PromptQueue - Prompt Admission flow: inbox -> promotion -> provider turn

Implements the full admission lifecycle for user prompts in DR-MMA:
  admit(prompt)   -> inbox (pending)
  promote()       -> atomic consume from inbox, status -> promoted
  drain(max_turns)-> loop promote + execute until no continuation

Thread-safe via threading.Lock. Zero external dependencies.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

# Prompt status constants
STATUS_PENDING = "pending"
STATUS_PROMOTED = "promoted"
STATUS_EXECUTING = "executing"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"


@dataclass
class AdmittedPrompt:
    """A prompt admitted into the queue, tracking its lifecycle status."""

    id: str = field(default_factory=lambda: f"PMT-{uuid.uuid4().hex[:8]}")
    session_id: str = ""
    content: str = ""
    priority: int = 0
    status: str = STATUS_PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    promoted_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdmittedPrompt":
        return cls(
            id=data.get("id", ""),
            session_id=data.get("session_id", ""),
            content=data.get("content", ""),
            priority=data.get("priority", 0),
            status=data.get("status", STATUS_PENDING),
            created_at=data.get("created_at", ""),
            promoted_at=data.get("promoted_at"),
        )


class PromptQueue:
    """In-memory prompt queue with admission, promotion, and drain.

    Flow:
      1. admit(prompt, session_id) -> inbox (pending)
      2. promote() -> atomic consume from inbox, status -> promoted
      3. drain(max_turns) -> loop promote + execute until no continuation

    Thread-safe via internal Lock. Priority scheduling: higher priority
    prompts are promoted first. Session isolation: each session's prompts
    are tracked separately.
    """

    def __init__(self):
        self._inbox: list[AdmittedPrompt] = []
        self._lock = threading.Lock()

    # -- Admission --

    def admit(
        self,
        prompt: str,
        session_id: str = "",
        priority: int = 0,
    ) -> AdmittedPrompt:
        """Admit a prompt into the inbox.

        Args:
            prompt: the prompt text content
            session_id: session identifier for isolation
            priority: scheduling priority (higher = promoted first)

        Returns:
            AdmittedPrompt with status=pending
        """
        admitted = AdmittedPrompt(
            content=prompt,
            session_id=session_id,
            priority=priority,
        )
        with self._lock:
            self._inbox.append(admitted)
        return admitted

    # -- Promotion --

    def promote(self) -> Optional[AdmittedPrompt]:
        """Atomically consume the highest-priority pending prompt from inbox.

        Returns the promoted prompt or None if inbox is empty or no pending
        prompts remain.
        """
        with self._lock:
            # Find highest-priority pending prompt
            best_idx = -1
            best_priority = -1
            for i, p in enumerate(self._inbox):
                if p.status == STATUS_PENDING and p.priority > best_priority:
                    best_idx = i
                    best_priority = p.priority

            if best_idx < 0:
                return None

            prompt = self._inbox.pop(best_idx)
            prompt.status = STATUS_PROMOTED
            prompt.promoted_at = datetime.now(timezone.utc).isoformat()
            return prompt

    # -- Peek --

    def peek_next(self, session_id: str = "") -> Optional[AdmittedPrompt]:
        """Look at the next pending prompt without consuming it.

        If session_id is given, only consider prompts from that session.
        Returns None if no matching pending prompt exists.
        """
        with self._lock:
            best: Optional[AdmittedPrompt] = None
            for p in self._inbox:
                if p.status != STATUS_PENDING:
                    continue
                if session_id and p.session_id != session_id:
                    continue
                if best is None or p.priority > best.priority:
                    best = p
            return best

    # -- Cancel --

    def cancel(self, prompt_id: str) -> bool:
        """Cancel a pending prompt by ID.

        Only pending prompts can be cancelled. Returns True if the prompt
        was found and cancelled, False otherwise.
        """
        with self._lock:
            for p in self._inbox:
                if p.id == prompt_id and p.status == STATUS_PENDING:
                    p.status = STATUS_CANCELLED
                    return True
            return False

    # -- Drain --

    def drain(
        self,
        max_turns: int = 1,
        executor: Optional[Callable[[AdmittedPrompt], dict]] = None,
        continuation_check: Optional[Callable[[dict], bool]] = None,
    ) -> list[dict]:
        """Process-local drain loop: promote + execute until no continuation.

        Promotes pending prompts one at a time, executes them via the
        executor callback, and checks for continuation. Stops when:
          - max_turns reached
          - no more pending prompts
          - continuation_check returns False for a result

        Args:
            max_turns: maximum number of promote+execute cycles
            executor: callable receiving AdmittedPrompt, returning result dict.
                     Must include key 'has_continuation' (bool) if
                     continuation_check is not provided.
            continuation_check: optional callable receiving result dict,
                               returning True to continue draining.
                               If omitted, uses result.get('has_continuation').

        Returns:
            List of result dicts from each turn.
        """
        _exec = executor
        if _exec is None:
            def default_executor(p: AdmittedPrompt) -> dict:
                return {"prompt_id": p.id, "content": p.content, "has_continuation": False}
            _exec = default_executor

        _check = continuation_check
        if _check is None:
            def default_check(result: dict) -> bool:
                return result.get("has_continuation", False)
            _check = default_check

        results: list[dict] = []
        turns = 0

        while turns < max_turns:
            prompt = self.promote()
            if prompt is None:
                break

            # Mark as executing before executor runs
            with self._lock:
                prompt.status = STATUS_EXECUTING

            result = _exec(prompt)

            # Mark as completed after successful execution
            with self._lock:
                prompt.status = STATUS_COMPLETED

            results.append(result)
            turns += 1

            if not _check(result):
                break

        return results

    # -- Diagnostics --

    def inbox_count(self, session_id: str = "") -> int:
        """Count pending prompts in the inbox.

        If session_id is given, only count prompts from that session.
        """
        with self._lock:
            count = 0
            for p in self._inbox:
                if p.status != STATUS_PENDING:
                    continue
                if session_id and p.session_id != session_id:
                    continue
                count += 1
            return count

    def queue_summary(self) -> dict:
        """Return a summary of the current queue state.

        Includes total inbox size, counts per status, and per-session
        breakdown.
        """
        with self._lock:
            status_counts: dict[str, int] = {}
            session_counts: dict[str, int] = {}

            for p in self._inbox:
                status_counts[p.status] = status_counts.get(p.status, 0) + 1
                if p.session_id:
                    session_counts[p.session_id] = (
                        session_counts.get(p.session_id, 0) + 1
                    )

            return {
                "total_inbox": len(self._inbox),
                "status_breakdown": status_counts,
                "session_breakdown": session_counts,
            }

    def get_by_id(self, prompt_id: str) -> Optional[AdmittedPrompt]:
        """Look up a prompt by its ID."""
        with self._lock:
            for p in self._inbox:
                if p.id == prompt_id:
                    return p
            return None

    def list_pending(self, session_id: str = "") -> list[AdmittedPrompt]:
        """List all pending prompts, optionally filtered by session."""
        with self._lock:
            results = []
            for p in self._inbox:
                if p.status != STATUS_PENDING:
                    continue
                if session_id and p.session_id != session_id:
                    continue
                results.append(p)
            return sorted(results, key=lambda x: -x.priority)

    def clear(self):
        """Clear the entire inbox."""
        with self._lock:
            self._inbox.clear()
