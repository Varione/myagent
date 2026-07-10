"""
SessionStore - SQLite Session Persistence Layer

Uses Python builtin sqlite3 with zero external dependencies. Provides full CRUD
for session / message / todo tables with relational queries and transactions.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class Session:
    """Session record"""

    def __init__(self, session_id: str = "", location: str = ""):
        self.session_id = session_id or f"SES-{uuid.uuid4().hex[:8]}"
        self.location = location
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = now
        self.updated_at = now

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "location": self.location,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        s = cls(session_id=data.get("session_id", ""), location=data.get("location", ""))
        s.created_at = data.get("created_at", s.created_at)
        s.updated_at = data.get("updated_at", s.updated_at)
        return s


class Message:
    """Message record"""

    def __init__(self, message_id: str = "", session_id: str = "",
                 role: str = "", content: str = ""):
        self.message_id = message_id or f"MSG-{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.role = role
        self.content = content
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        m = cls(message_id=data.get("message_id", ""),
                session_id=data.get("session_id", ""),
                role=data.get("role", ""), content=data.get("content", ""))
        m.timestamp = data.get("timestamp", m.timestamp)
        return m


class Todo:
    """Todo item"""

    def __init__(self, todo_id: str = "", session_id: str = "",
                 content: str = "", status: str = "pending", priority: int = 0):
        self.todo_id = todo_id or f"TODO-{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.content = content
        self.status = status
        self.priority = priority

    def to_dict(self) -> dict:
        return {
            "todo_id": self.todo_id,
            "session_id": self.session_id,
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Todo":
        return cls(
            todo_id=data.get("todo_id", ""),
            session_id=data.get("session_id", ""),
            content=data.get("content", ""),
            status=data.get("status", "pending"),
            priority=data.get("priority", 0),
        )


class SessionStore:
    """SQLite session persistence store

    Tables:
      session(id, location, created_at, updated_at)
      message(id, session_id, role, content, timestamp)
      todo(id, session_id, content, status, priority)
    """

    _CREATE_TABLES = """\
        CREATE TABLE IF NOT EXISTS session (
            id TEXT PRIMARY KEY,
            location TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS message (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES session(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS todo (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES session(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_message_session ON message(session_id);
        CREATE INDEX IF NOT EXISTS idx_todo_session ON todo(session_id);
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(self._CREATE_TABLES)
        self._conn.commit()
        self._is_open = True

    # -- Connection management --

    def close(self):
        """Close database connection"""
        if self._is_open and self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._is_open = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # -- Session CRUD --

    def create_session(self, location: str = "") -> Session:
        """Create a new session"""
        session = Session(location=location)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO session (id, location, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session.session_id, session.location, now, now),
        )
        self._conn.commit()
        session.created_at = now
        session.updated_at = now
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get single session by ID"""
        row = self._conn.execute(
            "SELECT id, location, created_at, updated_at FROM session WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return Session.from_dict(dict(zip(
            ["session_id", "location", "created_at", "updated_at"], row)))

    def list_sessions(self, limit: int = 0, offset: int = 0) -> list[Session]:
        """List all sessions ordered by creation time descending"""
        query = "SELECT id, location, created_at, updated_at FROM session ORDER BY created_at DESC"
        if limit > 0:
            query += f" LIMIT {limit} OFFSET {offset}"
        rows = self._conn.execute(query).fetchall()
        return [
            Session.from_dict(dict(zip(
                ["session_id", "location", "created_at", "updated_at"], row)))
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete session and all its messages and todos"""
        cursor = self._conn.execute("DELETE FROM session WHERE id = ?", (session_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count_sessions(self) -> int:
        """Count total sessions"""
        row = self._conn.execute("SELECT COUNT(*) FROM session").fetchone()
        return row[0] if row else 0

    # -- Message CRUD --

    def add_message(self, message_id: str, session_id: str,
                    role: str, content: str) -> Message:
        """Add a message to a session"""
        msg = Message(message_id=message_id, session_id=session_id,
                      role=role, content=content)
        self._conn.execute(
            "INSERT INTO message (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (msg.message_id, msg.session_id, msg.role, msg.content, msg.timestamp),
        )
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE session SET updated_at = ? WHERE id = ?", (now, session_id))
        self._conn.commit()
        return msg

    def get_messages(self, session_id: str, limit: int = 0,
                     offset: int = 0) -> list[Message]:
        """Get messages for a session ordered by timestamp ascending"""
        query = (
            "SELECT id, session_id, role, content, timestamp "
            "FROM message WHERE session_id = ? ORDER BY timestamp ASC"
        )
        if limit > 0:
            query += f" LIMIT {limit} OFFSET {offset}"
        rows = self._conn.execute(query, (session_id,)).fetchall()
        return [
            Message.from_dict(dict(zip(
                ["message_id", "session_id", "role", "content", "timestamp"], row)))
            for row in rows
        ]

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get single message by ID"""
        row = self._conn.execute(
            "SELECT id, session_id, role, content, timestamp FROM message WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        return Message.from_dict(dict(zip(
            ["message_id", "session_id", "role", "content", "timestamp"], row)))

    def delete_message(self, message_id: str) -> bool:
        """Delete single message"""
        cursor = self._conn.execute("DELETE FROM message WHERE id = ?", (message_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count_messages(self, session_id: str) -> int:
        """Count messages for a session"""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM message WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else 0

    # -- Todo CRUD --

    def add_todo(self, todo_id: str, session_id: str, content: str,
                 status: str = "pending", priority: int = 0) -> Todo:
        """Add a todo item"""
        todo = Todo(todo_id=todo_id, session_id=session_id,
                    content=content, status=status, priority=priority)
        self._conn.execute(
            "INSERT INTO todo (id, session_id, content, status, priority) VALUES (?, ?, ?, ?, ?)",
            (todo.todo_id, todo.session_id, todo.content, todo.status, todo.priority),
        )
        self._conn.commit()
        return todo

    def get_todos(self, session_id: str, status: str = "", limit: int = 0) -> list[Todo]:
        """Get todos for a session"""
        query = "SELECT id, session_id, content, status, priority FROM todo WHERE session_id = ?"
        params: list = [session_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY priority DESC"
        if limit > 0:
            query += f" LIMIT {limit}"
        rows = self._conn.execute(query, params).fetchall()
        return [
            Todo.from_dict(dict(zip(
                ["todo_id", "session_id", "content", "status", "priority"], row)))
            for row in rows
        ]

    def update_todo_status(self, todo_id: str, status: str) -> bool:
        """Update todo status"""
        cursor = self._conn.execute(
            "UPDATE todo SET status = ? WHERE id = ?", (status, todo_id))
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_todo(self, todo_id: str) -> bool:
        """Delete a todo"""
        cursor = self._conn.execute("DELETE FROM todo WHERE id = ?", (todo_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count_todos(self, session_id: str) -> int:
        """Count todos for a session"""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM todo WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else 0
