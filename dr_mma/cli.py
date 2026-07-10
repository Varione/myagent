"""
DR-MMA CLI - Command line interface for Dynamic Role-based Multi-Model Agent Architecture.

Provides subcommands for managing sessions, running agents, querying configuration,
and inspecting system state. Zero external dependencies (stdlib only).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cmd_version(args):
    """Print DR-MMA version information."""
    print("DR-MMA v0.1.0")
    print("Dynamic Role-based Multi-Model Agent Architecture")
    print(f"Python: {sys.version.split()[0]}")
    return 0


def cmd_sessions(args):
    """List or manage sessions via SessionStore."""
    from dr_mma.storage.session_store import SessionStore

    db_path = args.db or "data/sessions.db"
    store = SessionStore(db_path)

    try:
        if args.action == "list":
            sessions = store.list_sessions(limit=args.limit or 0)
            for s in sessions:
                print(f"  {s.session_id} | {s.location} | {s.created_at}")
            print(f"\nTotal: {len(sessions)} sessions")

        elif args.action == "create":
            session = store.create_session(location=args.location or "")
            print(json.dumps(session.to_dict(), indent=2))

        elif args.action == "delete":
            if store.delete_session(args.session_id):
                print(f"Deleted session: {args.session_id}")
            else:
                print(f"Session not found: {args.session_id}", file=sys.stderr)
                return 1
    finally:
        store.close()

    return 0


def cmd_messages(args):
    """View messages for a session."""
    from dr_mma.storage.session_store import SessionStore

    db_path = args.db or "data/sessions.db"
    store = SessionStore(db_path)

    try:
        messages = store.get_messages(args.session_id, limit=args.limit or 0)
        for m in messages:
            print(f"[{m.role}] {m.content[:100]}{'...' if len(m.content) > 100 else ''}")
        print(f"\nTotal: {len(messages)} messages")
    finally:
        store.close()

    return 0


def cmd_config(args):
    """Inspect configuration hierarchy."""
    from dr_mma.engine.config_hierarchy import ConfigHierarchy

    ch = ConfigHierarchy()

    if args.file:
        ch.load_project(args.file)

    merged = ch.merge()
    print(json.dumps(merged, indent=2, ensure_ascii=False))
    return 0


def cmd_models(args):
    """List available models in the model pool."""
    from dr_mma.engine.model_pool import ModelPool

    pool = ModelPool()

    if args.file:
        pool.load_from_file(args.file)
        print(f"Loaded models from: {args.file}")

    entries = pool.list_models()
    for e in entries:
        print(f"  {e.name} | provider={e.provider} | context={e.context_limit} | cost={e.cost_per_token}")
    print(f"\nTotal: {len(entries)} models")
    return 0


def cmd_tools(args):
    """List registered tools."""
    from dr_mma.engine.tools import ToolRegistry

    registry = ToolRegistry()

    if args.file:
        with open(args.file) as f:
            tools_data = json.load(f)
        for t in tools_data:
            registry.register(
                name=t["name"],
                description=t.get("description", ""),
                category=t.get("category", "general"),
            )

    tools = registry.list_tools()
    for t in tools:
        print(f"  {t.name} | {t.category} | safety={t.safety_level}")
    print(f"\nTotal: {len(tools)} tools")
    return 0


def cmd_stream(args):
    """Test streaming session."""
    from dr_mma.engine.streaming import StreamSession

    session = StreamSession(stream_id="cli-test")

    if args.text:
        for chunk in args.text.split(" "):
            session.send_chunk(chunk + " ")
        session.close()
        print(session.sse_output())
    else:
        session.send_chunk("DR-MMA streaming test OK\n")
        session.close()
        print(session.sse_output())

    return 0


def cmd_event_bus(args):
    """Test event bus publish/subscribe."""
    from dr_mma.engine.event_bus import EventBus

    bus = EventBus()

    if args.publish:
        event = bus.publish(args.publish, args.data or {})
        print(json.dumps(event.to_dict(), indent=2))

    history = bus.get_history()
    print(f"\nEvent history count: {len(history)}")
    return 0


def cmd_subagent(args):
    """Run a subagent task."""
    from dr_mma.engine.subagent_runner import SubAgentRunner

    with SubAgentRunner() as runner:
        handle = runner.spawn(args.prompt or "echo hello", args.agent or "test-agent")
        result = runner.run(handle)
        print(json.dumps(result.to_dict(), indent=2))

    return 0


def cmd_debate(args):
    """Run a mini debate between roles."""
    from dr_mma.engine.debate_room import DebateRoom, DebateTurn

    room = DebateRoom(
        topic=args.topic or "What is the best architecture?",
        max_rounds=args.rounds or 3,
    )

    roles = args.roles or ["proponent", "opponent"]
    for role in roles:
        room.add_participant(role)

    result = room.run()
    print(json.dumps({
        "topic": result.topic,
        "rounds": len(result.turns),
        "winner": result.winner,
    }, indent=2))
    return 0


def cmd_window(args):
    """Test window manager."""
    from dr_mma.engine.window_manager import WindowManager, WindowConfig

    config = WindowConfig(
        max_tokens=args.max_tokens or 4096,
        reserve_tokens=args.reserve or 512,
    )
    wm = WindowManager(config)

    if args.text:
        for line in args.text.split("\n"):
            wm.add_user(line.strip())

    snap = wm.snapshot()
    print(json.dumps({
        "total_tokens": snap.total_tokens,
        "message_count": len(snap.messages),
        "strategy": snap.strategy,
        "usage_ratio": round(wm.usage_ratio, 4),
    }, indent=2))
    return 0


def cmd_complexity(args):
    """Evaluate task complexity."""
    from dr_mma.engine.complexity import TaskComplexityEvaluator

    evaluator = TaskComplexityEvaluator()
    report = evaluator.evaluate(args.task or "implement a REST API")
    print(json.dumps(report.to_dict() if hasattr(report, 'to_dict') else str(report), indent=2))
    return 0


def cmd_capabilities(args):
    """List or calibrate agent capabilities."""
    from dr_mma.engine.capabilities import CapabilityRegistry

    registry = CapabilityRegistry()
    profiles = registry.list_profiles()
    for p in profiles:
        print(f"  {p.name} | domains={len(p.domains)} | level={p.level}")
    print(f"\nTotal: {len(profiles)} capability profiles")
    return 0


def cmd_budget(args):
    """Check or set budget constraints."""
    from dr_mma.engine.budget_controller import BudgetController

    task_id = "cli-budget"
    controller = BudgetController()
    controller.initialize(
        task_id=task_id,
        max_total_tokens=args.max_tokens or 100_000,
    )

    if args.use:
        controller.record_tokens(task_id, int(args.use))
        usage = controller.get_usage(task_id)
        remaining = (args.max_tokens or 100_000) - usage.tokens_consumed
        print(f"Consumed tokens, remaining: {remaining}")

    usage = controller.get_usage(task_id)
    print(json.dumps(usage.to_dict(), indent=2))
    return 0


def main(argv=None):
    """DR-MMA CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dr-mma",
        description="Dynamic Role-based Multi-Model Agent Architecture CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # sessions
    sess_p = subparsers.add_parser("sessions", help="Manage sessions")
    sess_p.add_argument("--db", default="data/sessions.db", help="Database path")
    sess_p.add_argument("--action", choices=["list", "create", "delete"], default="list")
    sess_p.add_argument("--session-id", dest="session_id", help="Session ID for delete")
    sess_p.add_argument("--location", help="Location for create")
    sess_p.add_argument("--limit", type=int, help="Max sessions to list")

    # messages
    msg_p = subparsers.add_parser("messages", help="View session messages")
    msg_p.add_argument("session_id", help="Session ID")
    msg_p.add_argument("--db", default="data/sessions.db", help="Database path")
    msg_p.add_argument("--limit", type=int, help="Max messages")

    # config
    cfg_p = subparsers.add_parser("config", help="Inspect configuration")
    cfg_p.add_argument("--file", help="Config file to load")

    # models
    mod_p = subparsers.add_parser("models", help="List available models")
    mod_p.add_argument("--file", help="Model pool JSON file")

    # tools
    tool_p = subparsers.add_parser("tools", help="List registered tools")
    tool_p.add_argument("--file", help="Tools JSON file")

    # stream
    stream_p = subparsers.add_parser("stream", help="Test streaming")
    stream_p.add_argument("--text", help="Text to stream")

    # event-bus
    eb_p = subparsers.add_parser("event-bus", help="Test event bus")
    eb_p.add_argument("--publish", help="Event type to publish")
    eb_p.add_argument("--data", default="{}", help="JSON data for event")

    # subagent
    sa_p = subparsers.add_parser("subagent", help="Run subagent task")
    sa_p.add_argument("--prompt", help="Prompt to execute")
    sa_p.add_argument("--agent", default="test-agent", help="Agent type")

    # debate
    db_p = subparsers.add_parser("debate", help="Run mini debate")
    db_p.add_argument("--topic", help="Debate topic")
    db_p.add_argument("--rounds", type=int, help="Max rounds")
    db_p.add_argument("--roles", nargs="+", help="Participant roles")

    # window
    win_p = subparsers.add_parser("window", help="Test window manager")
    win_p.add_argument("--max-tokens", type=int, help="Max tokens")
    win_p.add_argument("--reserve", type=int, help="Reserved tokens")
    win_p.add_argument("--text", help="Text to add to window")

    # complexity
    cx_p = subparsers.add_parser("complexity", help="Evaluate task complexity")
    cx_p.add_argument("--task", help="Task description")

    # capabilities
    cap_p = subparsers.add_parser("capabilities", help="List capabilities")

    # budget
    bud_p = subparsers.add_parser("budget", help="Check/set budget")
    bud_p.add_argument("--max-tokens", type=int, help="Max token budget")
    bud_p.add_argument("--max-cost", type=float, help="Max cost budget")
    bud_p.add_argument("--use", help="Consume tokens")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "version": cmd_version,
        "sessions": cmd_sessions,
        "messages": cmd_messages,
        "config": cmd_config,
        "models": cmd_models,
        "tools": cmd_tools,
        "stream": cmd_stream,
        "event-bus": cmd_event_bus,
        "subagent": cmd_subagent,
        "debate": cmd_debate,
        "window": cmd_window,
        "complexity": cmd_complexity,
        "capabilities": cmd_capabilities,
        "budget": cmd_budget,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
