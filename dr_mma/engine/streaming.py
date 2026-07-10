"""DR-MMA streaming engine -- SSE-style real-time response, tool-call progress, multi-consumer."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Event kind enumeration
# ---------------------------------------------------------------------------

class StreamEventKind(Enum):
    """Types of streaming events."""

    CHUNK = "chunk"             # Text increment
    TOOL_CALL = "tool_call"     # Tool invocation started
    TOOL_RESULT = "tool_result" # Tool invocation finished
    THINKING = "thinking"       # Chain-of-thought / reasoning step
    DONE = "done"               # Stream completed normally
    ERROR = "error"             # Stream terminated with error


# ---------------------------------------------------------------------------
# Event data class
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """Single event in a streaming session."""

    kind: StreamEventKind
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    seq: int = 0

    def to_sse(self) -> str:
        """Format as SSE block: ``event: <kind>\\ndata: <json>\\n\\n``"""
        return f"event: {self.kind.value}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "seq": self.seq,
        }


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class StreamProducer:
    """Push events to every subscribed consumer queue."""

    def __init__(self, stream_id: Optional[str] = None):
        self.stream_id: str = stream_id or uuid.uuid4().hex[:8]
        self._consumers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._seq: int = 0
        self._closed: bool = False
        self._events: list[StreamEvent] = []

    # -- subscription -------------------------------------------------------

    def subscribe(self, buffer_size: int = 100) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=buffer_size)
        with self._lock:
            self._consumers.append(q)
        return q

    # -- sending ------------------------------------------------------------

    def send(self, kind: StreamEventKind, data: Optional[dict] = None) -> StreamEvent:
        if self._closed:
            raise RuntimeError("StreamProducer already closed")
        self._seq += 1
        event = StreamEvent(kind=kind, data=data or {}, seq=self._seq)
        with self._lock:
            self._events.append(event)
            dead: list[queue.Queue] = []
            for q in self._consumers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._consumers.remove(q)
        return event

    def send_chunk(self, text: str) -> StreamEvent:
        return self.send(StreamEventKind.CHUNK, {"text": text})

    def send_tool_call(self, tool_name: str, args: dict) -> StreamEvent:
        return self.send(StreamEventKind.TOOL_CALL, {"tool": tool_name, "args": args})

    def send_tool_result(self, tool_name: str, result: dict, success: bool = True) -> StreamEvent:
        return self.send(StreamEventKind.TOOL_RESULT, {
            "tool": tool_name, "result": result, "success": success,
        })

    def send_thinking(self, content: str) -> StreamEvent:
        return self.send(StreamEventKind.THINKING, {"content": content})

    # -- close --------------------------------------------------------------

    def close(self, error: Optional[str] = None) -> list[StreamEvent]:
        if error:
            self.send(StreamEventKind.ERROR, {"error": error})
        else:
            self.send(StreamEventKind.DONE, {})
        self._closed = True
        return list(self._events)

    # -- properties ---------------------------------------------------------

    @property
    def history(self) -> list[StreamEvent]:
        return list(self._events)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def consumer_count(self) -> int:
        with self._lock:
            return len(self._consumers)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

class StreamConsumer:
    """Pull events from a queue; supports timeout and kind filtering."""

    def __init__(self, q: queue.Queue, producer: StreamProducer):
        self._queue = q
        self._producer = producer
        self._buffer: list[StreamEvent] = []
        self._filters: Optional[set[StreamEventKind]] = None

    def set_filter(self, kinds: set[StreamEventKind]) -> None:
        self._filters = kinds

    def receive(self, timeout: Optional[float] = None) -> Optional[StreamEvent]:
        try:
            event = self._queue.get(timeout=timeout)
            if self._filters is None or event.kind in self._filters:
                self._buffer.append(event)
                return event
            return None
        except queue.Empty:
            return None

    def receive_all(self, timeout: float = 1.0) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        while True:
            event = self.receive(timeout=timeout)
            if event is None:
                break
            events.append(event)
        return events

    @property
    def buffer(self) -> list[StreamEvent]:
        return list(self._buffer)

    @property
    def chunks(self) -> str:
        return "".join(
            e.data.get("text", "") for e in self._buffer if e.kind == StreamEventKind.CHUNK
        )

    @property
    def is_done(self) -> bool:
        return any(
            e.kind in (StreamEventKind.DONE, StreamEventKind.ERROR)
            for e in self._buffer
        )


# ---------------------------------------------------------------------------
# Session  (high-level wrapper)
# ---------------------------------------------------------------------------

class StreamSession:
    """Manage one streaming session: producer + consumers + callbacks."""

    def __init__(self, stream_id: Optional[str] = None):
        self._producer = StreamProducer(stream_id)
        self._consumers: list[StreamConsumer] = []
        self._callbacks: list[Callable[[StreamEvent], None]] = []

    @property
    def producer(self) -> StreamProducer:
        return self._producer

    @property
    def stream_id(self) -> str:
        return self._producer.stream_id

    # -- consumers ----------------------------------------------------------

    def add_consumer(self, buffer_size: int = 100) -> StreamConsumer:
        q = self._producer.subscribe(buffer_size)
        consumer = StreamConsumer(q, self._producer)
        self._consumers.append(consumer)
        return consumer

    # -- callbacks ----------------------------------------------------------

    def on_event(self, callback: Callable[[StreamEvent], None]) -> None:
        self._callbacks.append(callback)

    # -- sending ------------------------------------------------------------

    def send(self, kind: StreamEventKind, data: Optional[dict] = None) -> StreamEvent:
        event = self._producer.send(kind, data)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass
        return event

    def send_chunk(self, text: str) -> StreamEvent:
        return self.send(StreamEventKind.CHUNK, {"text": text})

    def send_tool_call(self, tool_name: str, args: dict) -> StreamEvent:
        return self.send(StreamEventKind.TOOL_CALL, {"tool": tool_name, "args": args})

    def send_tool_result(self, tool_name: str, result: dict, success: bool = True) -> StreamEvent:
        return self.send(StreamEventKind.TOOL_RESULT, {
            "tool": tool_name, "result": result, "success": success,
        })

    def send_thinking(self, content: str) -> StreamEvent:
        return self.send(StreamEventKind.THINKING, {"content": content})

    # -- close --------------------------------------------------------------

    def close(self, error: Optional[str] = None) -> list[StreamEvent]:
        if error:
            event = self.send(StreamEventKind.ERROR, {"error": error})
        else:
            event = self.send(StreamEventKind.DONE, {})
        self._producer._closed = True
        return list(self._producer._events)

    # -- helpers ------------------------------------------------------------

    def sse_output(self) -> str:
        return "".join(e.to_sse() for e in self._producer.history)
