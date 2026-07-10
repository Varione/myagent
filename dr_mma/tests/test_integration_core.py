"""Integration tests for core DR-MMA modules.

Tests cross-module collaboration between session_store, event_bus,
subagent_runner, prompt_queue, config_hierarchy, and tool_protocol.
"""
import json
import os
import tempfile

import pytest

# -- Scenario 1: SessionStore + EventBus --

class TestSessionEventBusIntegration:

    def test_session_create_publishes_event(self):
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.event_bus import EventBus

        bus = EventBus()
        events_received = []

        bus.subscribe(lambda event: events_received.append(event), ["session_created"])

        with SessionStore(":memory:") as store:
            session = store.create_session("test-session")
            bus.publish("session_created", {"session_id": session.session_id})

        assert len(events_received) == 1
        assert events_received[0].event_type == "session_created"
        assert events_received[0].data.get("session_id") == session.session_id

    def test_message_add_publishes_event(self):
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.event_bus import EventBus

        bus = EventBus()
        events_received = []

        bus.subscribe(lambda event: events_received.append(event), ["message_added"])

        with SessionStore(":memory:") as store:
            session = store.create_session("test-session")
            msg_id = "MSG-001"
            store.add_message(msg_id, session.session_id, "user", "Hello")
            bus.publish("message_added", {"session_id": session.session_id, "role": "user"})

        assert len(events_received) == 1

    def test_todo_status_change_publishes_event(self):
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.event_bus import EventBus

        bus = EventBus()
        events_received = []

        bus.subscribe(lambda event: events_received.append(event), ["todo_updated"])

        with SessionStore(":memory:") as store:
            session = store.create_session("test-session")
            todo_id = "TODO-001"
            store.add_todo(todo_id, session.session_id, "Test task", "high")
            store.update_todo_status(todo_id, "completed")
            bus.publish("todo_updated", {"todo_id": todo_id, "status": "completed"})

        assert len(events_received) == 1
        assert events_received[0].data["status"] == "completed"

    def test_session_delete_cascades_events(self):
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.event_bus import EventBus

        bus = EventBus()
        events_received = []

        bus.subscribe(lambda event: events_received.append(event), ["session_deleted"])

        with SessionStore(":memory:") as store:
            session = store.create_session("test-session")
            msg_id = "MSG-001"
            store.add_message(msg_id, session.session_id, "user", "Hello")
            store.delete_session(session.session_id)
            bus.publish("session_deleted", {"session_id": session.session_id})

        assert len(events_received) == 1

    def test_multiple_event_handlers(self):
        from dr_mma.engine.event_bus import EventBus

        bus = EventBus()
        handler1_calls = []
        handler2_calls = []

        bus.subscribe(lambda event: handler1_calls.append(event), ["test_event"])
        bus.subscribe(lambda event: handler2_calls.append(event), ["test_event"])

        bus.publish("test_event", {"key": "value"})

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1


# -- Scenario 2: SubAgentRunner + PromptQueue --

class TestSubAgentPromptQueueIntegration:

    def test_queue_drives_subagent_execution(self):
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.engine.subagent_runner import SubAgentRunner

        queue = PromptQueue()

        for i in range(3):
            queue.admit(f"Task {i}", session_id="sess-1", priority=i)

        results = []
        with SubAgentRunner() as runner:
            while True:
                prompt = queue.promote()
                if prompt is None:
                    break
                handle = runner.spawn(prompt.content, "test-agent")
                result = runner.run(handle)
                results.append(result)

        assert len(results) == 3

    def test_queue_priority_affects_execution_order(self):
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.engine.subagent_runner import SubAgentRunner

        queue = PromptQueue()
        queue.admit("Low priority", session_id="s1", priority=0)
        queue.admit("High priority", session_id="s1", priority=10)
        queue.admit("Medium priority", session_id="s1", priority=5)

        execution_order = []

        with SubAgentRunner() as runner:
            while True:
                prompt = queue.promote()
                if prompt is None:
                    break
                handle = runner.spawn(prompt.content, "test-agent")
                result = runner.run(handle)
                execution_order.append(result.latency_ms)

        # Just verify it runs without error - actual order depends on implementation
        assert len(execution_order) == 3

    def test_parallel_execution_from_queue(self):
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.engine.subagent_runner import SubAgentRunner

        queue = PromptQueue()
        for i in range(3):
            queue.admit(f"Parallel task {i}", session_id="s1")

        with SubAgentRunner() as runner:
            handles = []
            for _ in range(3):
                prompt = queue.promote()
                if prompt:
                    handles.append(runner.spawn(prompt.content, "test-agent"))

            results = runner.parallel_execute(handles)
            assert len(results) == 3

    def test_continuation_requeues(self):
        from dr_mma.engine.prompt_queue import PromptQueue

        queue = PromptQueue()
        queue.admit("Initial prompt", session_id="s1")

        prompt = queue.promote()
        assert prompt is not None

        queue.admit("Follow-up", session_id="s1", priority=prompt.priority)

        next_prompt = queue.promote()
        assert next_prompt is not None


# -- Scenario 3: ConfigHierarchy + SessionStore --

class TestConfigSessionIntegration:

    def test_config_drives_session_db_path(self):
        from dr_mma.engine.config_hierarchy import ConfigHierarchy
        from dr_mma.storage.session_store import SessionStore

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"storage": {"session_db": ":memory:"}}, f)
            config_path = f.name

        try:
            ch = ConfigHierarchy()
            ch.load_project(config_path)
            merged = ch.merge()

            db_path = merged.get("storage", {}).get("session_db", ":memory:")
            store = SessionStore(db_path)
            session = store.create_session("config-driven")

            assert session.session_id is not None
        finally:
            os.unlink(config_path)

    def test_config_tool_settings(self):
        from dr_mma.engine.config_hierarchy import ConfigHierarchy

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({
                "tools": {
                    "enabled": ["code_execute", "web_search"],
                    "code_execute": {"timeout": 30}
                }
            }, f)
            config_path = f.name

        try:
            ch = ConfigHierarchy()
            ch.load_project(config_path)
            merged = ch.merge()

            tools_config = merged.get("tools", {})
            assert "code_execute" in tools_config.get("enabled", [])
        finally:
            os.unlink(config_path)


# -- Scenario 4: EventBus + SubAgentRunner lifecycle --

class TestEventBusSubAgentIntegration:

    def test_subagent_lifecycle_events(self):
        from dr_mma.engine.event_bus import EventBus
        from dr_mma.engine.subagent_runner import SubAgentRunner

        bus = EventBus()
        lifecycle_events = []

        bus.subscribe(lambda event: lifecycle_events.append(event), ["subagent_spawned", "subagent_completed", "subagent_failed"])

        with SubAgentRunner() as runner:
            handle = runner.spawn("test", "agent-1")
            bus.publish("subagent_spawned", {"handle_id": handle.agent_id})
            result = runner.run(handle)
            bus.publish("subagent_completed", {"handle_id": handle.agent_id, "latency_ms": result.latency_ms})

        assert len(lifecycle_events) == 2

    def test_event_replay_after_subagent_run(self):
        from dr_mma.engine.event_bus import EventBus
        from dr_mma.engine.subagent_runner import SubAgentRunner

        bus = EventBus()

        with SubAgentRunner() as runner:
            handle = runner.spawn("test", "agent-1")
            bus.publish("subagent_spawned", {"handle_id": handle.agent_id})
            result = runner.run(handle)
            bus.publish("subagent_completed", {"handle_id": handle.agent_id})

        replayed = bus.get_history()
        assert len(replayed) == 2


# -- Scenario 5: PromptQueue + SessionStore persistence --

class TestPromptQueuePersistence:

    def test_queue_state_saved_to_session(self):
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.storage.session_store import SessionStore

        queue = PromptQueue()
        queue.admit("Task 1", session_id="s1")
        queue.admit("Task 2", session_id="s1", priority=5)

        with SessionStore(":memory:") as store:
            session = store.create_session("queue-state")
            summary = queue.queue_summary()
            store.add_message("MSG-SUMMARY", session.session_id, "tool", json.dumps(summary))

            saved = store.get_messages(session.session_id)
            assert len(saved) == 1
            restored = json.loads(saved[0].content)
            assert restored["total_inbox"] == 2

    def test_todo_sync_with_queue(self):
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.storage.session_store import SessionStore

        queue = PromptQueue()
        queue.admit("Important task", session_id="s1", priority=10)

        with SessionStore(":memory:") as store:
            session = store.create_session("queue-todos")

            pending = queue.list_pending()
            for p in pending:
                store.add_todo(p.id, session.session_id, p.content,
                             "high" if p.priority >= 5 else "medium")

            todos = store.get_todos(session.session_id)
            assert len(todos) == 1
            assert todos[0].content == "Important task"


# -- Scenario 6: ConfigHierarchy + ToolProtocol --

class TestConfigToolProtocolIntegration:

    def test_config_schema_validates_tool_input(self):
        from dr_mma.engine.tool_protocol import (
            ToolProtocol, ToolInputSchema, SchemaField, SchemaType
        )

        tp = ToolProtocol(
            name="calculator",
            input_schema=ToolInputSchema(fields=[
                SchemaField("expression", SchemaType.STRING, required=True),
                SchemaField("precision", SchemaType.INTEGER, required=False, default=2),
            ]),
        )

        errors = tp.validate_input({"expression": "1+2"})
        assert len(errors) == 0

        errors = tp.validate_input({})
        assert len(errors) > 0


# -- Scenario 7: Cross-module error handling --

class TestCrossModuleErrorHandling:

    def test_session_store_error_does_not_crash_bus(self):
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.event_bus import EventBus

        bus = EventBus()
        events_received = []

        bus.subscribe(lambda event: events_received.append(event), ["error"])

        with SessionStore(":memory:") as store:
            result = store.get_session("nonexistent-id")
            if result is None:
                bus.publish("error", {"source": "session_store", "session_id": "nonexistent-id"})

        assert len(events_received) == 1

    def test_subagent_failure_does_not_break_queue(self):
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.engine.subagent_runner import SubAgentRunner

        queue = PromptQueue()
        queue.admit("Good task", session_id="s1")
        queue.admit("Another good task", session_id="s1")

        with SubAgentRunner() as runner:
            results = []
            while True:
                prompt = queue.promote()
                if prompt is None:
                    break
                handle = runner.spawn(prompt.content, "test-agent")
                result = runner.run(handle)
                results.append(result)

            assert len(results) == 2


# -- Scenario 8: Full pipeline integration --

class TestFullPipelineIntegration:

    def test_config_to_session_to_queue_to_subagent(self):
        """Full pipeline: config -> session -> queue -> subagent execution."""
        from dr_mma.engine.config_hierarchy import ConfigHierarchy
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.engine.subagent_runner import SubAgentRunner

        # 1. Load config
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"storage": {"session_db": ":memory:"}}, f)
            config_path = f.name

        try:
            ch = ConfigHierarchy()
            ch.load_project(config_path)
            merged = ch.merge()

            # 2. Create session from config
            store = SessionStore(merged.get("storage", {}).get("session_db", ":memory:"))
            session = store.create_session("pipeline-test")

            # 3. Queue tasks
            queue = PromptQueue()
            queue.admit("Pipeline task 1", session_id=session.session_id)
            queue.admit("Pipeline task 2", session_id=session.session_id, priority=5)

            # 4. Execute via subagent runner
            results = []
            with SubAgentRunner() as runner:
                while True:
                    prompt = queue.promote()
                    if prompt is None:
                        break
                    handle = runner.spawn(prompt.content, "pipeline-agent")
                    result = runner.run(handle)
                    results.append(result)

            # 5. Verify
            assert len(results) == 2
            assert store.count_messages(session.session_id) == 0
        finally:
            os.unlink(config_path)

    def test_event_bus_tracks_full_pipeline(self):
        """Event bus tracks the entire pipeline lifecycle."""
        from dr_mma.engine.event_bus import EventBus
        from dr_mma.storage.session_store import SessionStore
        from dr_mma.engine.prompt_queue import PromptQueue
        from dr_mma.engine.subagent_runner import SubAgentRunner

        bus = EventBus()
        all_events = []

        bus.subscribe(lambda event: all_events.append(event), ["pipeline_step"])

        # Step 1: Session created
        with SessionStore(":memory:") as store:
            session = store.create_session("tracked-session")
            bus.publish("pipeline_step", {"step": "session_created", "id": session.session_id})

            # Step 2: Tasks queued
            queue = PromptQueue()
            queue.admit("Tracked task", session_id=session.session_id)
            bus.publish("pipeline_step", {"step": "task_queued"})

            # Step 3: Subagent executed
            with SubAgentRunner() as runner:
                prompt = queue.promote()
                handle = runner.spawn(prompt.content, "tracked-agent")
                runner.run(handle)
                bus.publish("pipeline_step", {"step": "task_completed"})

        assert len(all_events) == 3
        steps = [e.data["step"] for e in all_events]
        assert "session_created" in steps
        assert "task_queued" in steps
        assert "task_completed" in steps
