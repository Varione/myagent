"""SessionStore persistence layer tests."""

import os
import tempfile
import time
import pytest
from dr_mma.storage.session_store import Session, Message, Todo, SessionStore


@pytest.fixture
def tmp_db():
    """Provide a temporary database path and clean up after test"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_session.db")
        yield db_path


@pytest.fixture
def store(tmp_db):
    """Provide a SessionStore instance and close it after test"""
    s = SessionStore(tmp_db)
    yield s
    s.close()


# -- Session model tests --

class TestSession:
    def test_auto_generate_id(self):
        s = Session()
        assert s.session_id.startswith("SES-")
        assert len(s.session_id) == 12

    def test_custom_id(self):
        s = Session(session_id="custom-123")
        assert s.session_id == "custom-123"

    def test_location_stored(self):
        s = Session(location="/home/user/project")
        assert s.location == "/home/user/project"

    def test_timestamps_set(self):
        s = Session()
        assert s.created_at
        assert s.updated_at
        assert s.created_at == s.updated_at

    def test_to_dict_roundtrip(self):
        s = Session(session_id="S1", location="/loc")
        d = s.to_dict()
        s2 = Session.from_dict(d)
        assert s2.session_id == "S1"
        assert s2.location == "/loc"


# -- Message model tests --

class TestMessage:
    def test_auto_generate_id(self):
        m = Message()
        assert m.message_id.startswith("MSG-")

    def test_custom_id(self):
        m = Message(message_id="m-custom")
        assert m.message_id == "m-custom"

    def test_to_dict_roundtrip(self):
        m = Message(message_id="M1", session_id="S1", role="user", content="hello")
        d = m.to_dict()
        m2 = Message.from_dict(d)
        assert m2.message_id == "M1"
        assert m2.role == "user"


# -- Todo model tests --

class TestTodo:
    def test_auto_generate_id(self):
        t = Todo()
        assert t.todo_id.startswith("TODO-")

    def test_default_status(self):
        t = Todo()
        assert t.status == "pending"

    def test_default_priority(self):
        t = Todo()
        assert t.priority == 0

    def test_to_dict_roundtrip(self):
        t = Todo(todo_id="T1", session_id="S1", content="do stuff", status="done", priority=5)
        d = t.to_dict()
        t2 = Todo.from_dict(d)
        assert t2.status == "done"
        assert t2.priority == 5


# -- SessionStore CRUD tests --

class TestSessionStoreSession:
    def test_create_session(self, store):
        s = store.create_session(location="/test")
        assert s.session_id
        assert s.location == "/test"
        assert store.count_sessions() == 1

    def test_create_multiple_sessions(self, store):
        store.create_session("loc1")
        store.create_session("loc2")
        assert store.count_sessions() == 2

    def test_get_existing_session(self, store):
        s = store.create_session("/find-me")
        found = store.get_session(s.session_id)
        assert found is not None
        assert found.location == "/find-me"

    def test_get_nonexistent_session(self, store):
        assert store.get_session("nonexistent") is None

    def test_list_sessions_order(self, store):
        s1 = store.create_session("first")
        time.sleep(0.02)
        s2 = store.create_session("second")
        sessions = store.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].session_id == s2.session_id

    def test_list_sessions_limit(self, store):
        for i in range(5):
            store.create_session(f"loc-{i}")
        limited = store.list_sessions(limit=2)
        assert len(limited) == 2

    def test_list_sessions_offset(self, store):
        for i in range(4):
            store.create_session(f"loc-{i}")
        offset_results = store.list_sessions(limit=2, offset=2)
        assert len(offset_results) == 2

    def test_delete_session(self, store):
        s = store.create_session()
        assert store.delete_session(s.session_id) is True
        assert store.count_sessions() == 0

    def test_delete_nonexistent_session(self, store):
        assert store.delete_session("no-such-id") is False

    def test_context_manager(self, tmp_db):
        with SessionStore(tmp_db) as s:
            s.create_session()
            assert s.count_sessions() == 1

    def test_persistence_across_instances(self, tmp_db):
        with SessionStore(tmp_db) as s1:
            session = s1.create_session("persisted")
        with SessionStore(tmp_db) as s2:
            found = s2.get_session(session.session_id)
            assert found is not None
            assert found.location == "persisted"


# -- SessionStore Message tests --

class TestSessionStoreMessage:
    def test_add_message(self, store):
        s = store.create_session()
        m = store.add_message("m1", s.session_id, "user", "hello")
        assert m.message_id == "m1"
        assert m.role == "user"

    def test_get_messages(self, store):
        s = store.create_session()
        store.add_message("m1", s.session_id, "user", "first")
        store.add_message("m2", s.session_id, "assistant", "second")
        msgs = store.get_messages(s.session_id)
        assert len(msgs) == 2
        assert msgs[0].content == "first"

    def test_get_messages_empty_session(self, store):
        s = store.create_session()
        assert store.get_messages(s.session_id) == []

    def test_get_messages_limit(self, store):
        s = store.create_session()
        for i in range(5):
            store.add_message(f"m{i}", s.session_id, "user", f"msg-{i}")
        limited = store.get_messages(s.session_id, limit=2)
        assert len(limited) == 2

    def test_get_single_message(self, store):
        s = store.create_session()
        store.add_message("m-single", s.session_id, "user", "test")
        found = store.get_message("m-single")
        assert found is not None
        assert found.content == "test"

    def test_get_nonexistent_message(self, store):
        assert store.get_message("no-msg") is None

    def test_delete_message(self, store):
        s = store.create_session()
        store.add_message("m-del", s.session_id, "user", "bye")
        assert store.delete_message("m-del") is True
        assert store.count_messages(s.session_id) == 0

    def test_delete_nonexistent_message(self, store):
        assert store.delete_message("no-msg") is False

    def test_count_messages(self, store):
        s = store.create_session()
        store.add_message("m1", s.session_id, "user", "a")
        store.add_message("m2", s.session_id, "assistant", "b")
        assert store.count_messages(s.session_id) == 2

    def test_session_updated_at_on_message(self, store):
        s = store.create_session()
        old_updated = s.updated_at
        time.sleep(0.05)
        store.add_message("m1", s.session_id, "user", "update")
        new_s = store.get_session(s.session_id)
        assert new_s.updated_at != old_updated

    def test_messages_isolated_by_session(self, store):
        s1 = store.create_session()
        s2 = store.create_session()
        store.add_message("m1", s1.session_id, "user", "in s1")
        store.add_message("m2", s2.session_id, "user", "in s2")
        assert len(store.get_messages(s1.session_id)) == 1
        assert len(store.get_messages(s2.session_id)) == 1


# -- SessionStore Todo tests --

class TestSessionStoreTodo:
    def test_add_todo(self, store):
        s = store.create_session()
        t = store.add_todo("t1", s.session_id, "buy milk")
        assert t.todo_id == "t1"
        assert t.status == "pending"

    def test_get_todos(self, store):
        s = store.create_session()
        store.add_todo("t1", s.session_id, "low", priority=1)
        store.add_todo("t2", s.session_id, "high", priority=10)
        todos = store.get_todos(s.session_id)
        assert len(todos) == 2
        assert todos[0].priority == 10

    def test_get_todos_by_status(self, store):
        s = store.create_session()
        store.add_todo("t1", s.session_id, "a", status="pending")
        store.add_todo("t2", s.session_id, "b", status="done")
        pending = store.get_todos(s.session_id, status="pending")
        assert len(pending) == 1

    def test_update_todo_status(self, store):
        s = store.create_session()
        store.add_todo("t1", s.session_id, "task")
        assert store.update_todo_status("t1", "done") is True
        todos = store.get_todos(s.session_id, status="done")
        assert len(todos) == 1

    def test_update_nonexistent_todo(self, store):
        assert store.update_todo_status("no-todo", "done") is False

    def test_delete_todo(self, store):
        s = store.create_session()
        store.add_todo("t1", s.session_id, "task")
        assert store.delete_todo("t1") is True
        assert store.count_todos(s.session_id) == 0

    def test_count_todos(self, store):
        s = store.create_session()
        store.add_todo("t1", s.session_id, "a")
        store.add_todo("t2", s.session_id, "b")
        assert store.count_todos(s.session_id) == 2

    def test_cascade_delete_session_removes_messages(self, store):
        s = store.create_session()
        store.add_message("m1", s.session_id, "user", "hello")
        store.add_todo("t1", s.session_id, "task")
        store.delete_session(s.session_id)
        assert store.count_sessions() == 0
        assert store.count_messages(s.session_id) == 0
        assert store.count_todos(s.session_id) == 0

    def test_get_todos_limit(self, store):
        s = store.create_session()
        for i in range(10):
            store.add_todo(f"t{i}", s.session_id, f"task-{i}", priority=i)
        limited = store.get_todos(s.session_id, limit=3)
        assert len(limited) == 3
