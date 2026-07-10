# DR-MMA

Dynamic Role-based Multi-Model Agent Architecture — structured contract-driven multi-model review loop with zero external dependencies.

## Overview

DR-MMA is a multi-agent orchestration framework that dynamically assigns roles, routes tasks by complexity, and coordinates multiple LLM models through a structured workflow engine. Built entirely on Python stdlib (plus CustomTkinter for the optional GUI), with 1267 passing tests and zero regressions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Workflow Engine                         │
│  Task → Complexity Analysis → Mode Selection → Execution   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Planner  │  │ Executor │  │ Researcher│  │ Verifier │   │
│  │ (Supervisor Roles)                                          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Domain   │  │ Debate   │  │ Budget   │  │ Context  │   │
│  │ Agents   │  │ Room     │  │ Control  │  │ Manager  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model Pool │ Role Manager │ Capability Registry     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### Engine (26 modules)

| Module | Description |
|--------|-------------|
| `workflow.py` | Main orchestration engine with complexity routing and event logging |
| `streaming.py` | SSE-style streaming with multi-consumer broadcast and callback support |
| `window_manager.py` | Token-aware context window with importance-based trimming |
| `subagent_runner.py` | Sub-agent lifecycle management with DAG parallel execution |
| `debate_room.py` | Multi-participant debate for conflicting decisions |
| `supervisor_modules.py` | Task understanding, DAG planning, event handling, decision making, final review |
| `model_pool.py` | Multi-model registry with health tracking |
| `role_manager.py` | Dynamic role assignment, merge/split, and failover |
| `capabilities.py` | Capability profiles and dynamic role assigner |
| `capability_calibrator.py` | Agent capability calibration and status tracking |
| `complexity.py` | Task complexity evaluation with mode selection (direct/standard/review/expanded) |
| `budget_controller.py` | Multi-dimensional budget tracking (tokens, calls, debates, retries) |
| `permissions.py` | Role-based permission matrix with action levels and audit logging |
| `context_manager.py` | Runtime, global, and artifact context management |
| `context_epoch.py` | Epoch-based context snapshots and baseline management |
| `compaction.py` | Sliding window compaction with configurable triggers |
| `event_bus.py` | In-memory pub/sub event stream with history replay and SSE output |
| `prompt_queue.py` | Priority-based prompt queue with session isolation |
| `config_hierarchy.py` | Multi-layer configuration (system/project/user/env) with validation |
| `mcp_client.py` | MCP protocol client for tool discovery and invocation |
| `tool_protocol.py` | Schema validation, materialize/settle protocol for tools |
| `tools.py` | Tool registry with batch calling and usage tracking |
| `vector_memory.py` | TF-IDF vector memory with similarity search and eviction |
| `observability.py` | Event tracing, DAG visualization, and diagnostics panel |
| `domain_agents.py` | Domain agent registry with calibration status |

### Storage (5 modules)

| Module | Description |
|--------|-------------|
| `session_store.py` | SQLite session/message/todo persistence with transactions |
| `blackboard.py` | Shared blackboard for inter-agent communication |
| `decision_log.py` | Persistent decision audit trail |
| `artifact_store.py` | File-based artifact storage |

### Domains (6 agents)

| Domain | Capabilities |
|--------|-------------|
| `code_dev` | Code generation, review, debugging, refactoring |
| `paper_writing` | Literature review, abstract writing, format proofing |
| `data_analysis` | Statistical analysis, visualization, reporting |
| `engineering_sim` | Simulation design, parameter optimization, result analysis |
| `knowledge_mgmt` | Knowledge extraction, organization, retrieval |

### UI (12 modules)

CustomTkinter-based desktop application with chat panel, task pipeline, configuration, logging, and results visualization.

## Quick Start

### Installation

```bash
# Python 3.12+ required, stdlib only (no pip install needed for core)
cd dr_mma
python -m pytest tests/  # verify installation
```

### CLI Usage

```bash
python -m cli version
python -m cli sessions --action list
python -m cli config --file config.json
python -m cli stream --text "hello world"
python -m cli subagent --prompt "analyze this" --agent researcher
python -m cli window --max-tokens 4096 --text "your context here"
python -m cli budget --max-tokens 100000 --use 5000
```

### Programmatic Usage

```python
from dr_mma.engine.streaming import StreamSession
from dr_mma.engine.window_manager import WindowManager, WindowConfig
from dr_mma.engine.subagent_runner import SubAgentRunner
from dr_mma.storage.session_store import SessionStore

# Streaming
session = StreamSession(stream_id="my-stream")
session.send_chunk("Hello ")
session.send_chunk("World")
session.close()
print(session.sse_output())

# Window management
wm = WindowManager(WindowConfig(max_tokens=4096, reserve_tokens=512))
wm.add_user("What is quantum computing?")
wm.add_assistant("Quantum computing uses qubits...")
print(f"Usage: {wm.usage_ratio:.2%}")

# Sub-agent execution
with SubAgentRunner() as runner:
    handle = runner.spawn("Research topic", "researcher")
    result = runner.run(handle)
    print(result.content)

# Session persistence
with SessionStore(":memory:") as store:
    session = store.create_session("my-session")
    store.add_message("msg-1", session.session_id, "user", "Hello")
```

## Workflow Modes

The complexity evaluator automatically selects execution mode based on task analysis:

| Mode | When Used | Characteristics |
|------|-----------|-----------------|
| `direct` | Simple, single-step tasks | Single model call, no review |
| `standard` | Moderate complexity | Plan → Execute → Verify |
| `single_review` | Requires quality check | Standard + critic review |
| `expanded` | High complexity, multi-domain | Full debate room, multi-agent collaboration |

## Testing

```bash
# Full regression (1267 tests)
python -m pytest dr_mma/tests/ -v

# Specific module
python -m pytest dr_mma/tests/test_streaming.py -v

# Integration tests only
python -m pytest dr_mma/tests/test_integration_*.py -v
```

## Project Structure

```
dr_mma/
├── __init__.py
├── cli.py                    # CLI entry point (14 subcommands)
├── engine/                   # Core orchestration engine (26 modules)
│   ├── __init__.py           # Full module exports
│   ├── workflow.py           # Main workflow engine
│   ├── streaming.py          # SSE streaming
│   ├── window_manager.py     # Context window management
│   ├── subagent_runner.py    # Sub-agent lifecycle
│   ├── debate_room.py        # Multi-agent debate
│   ├── supervisor_modules.py # Planning, decisions, review
│   ├── model_pool.py         # Model registry
│   ├── role_manager.py       # Dynamic role assignment
│   ├── capabilities.py       # Capability profiles
│   ├── complexity.py         # Task complexity evaluation
│   ├── budget_controller.py  # Budget tracking
│   ├── permissions.py        # Role-based permissions
│   ├── context_manager.py    # Context management
│   ├── event_bus.py          # Pub/sub events
│   ├── prompt_queue.py       # Priority queue
│   ├── config_hierarchy.py   # Configuration layers
│   ├── mcp_client.py         # MCP protocol client
│   ├── tool_protocol.py      # Tool schema validation
│   ├── tools.py              # Tool registry
│   ├── vector_memory.py      # Vector search memory
│   ├── observability.py      # Tracing and diagnostics
│   ├── domain_agents.py      # Domain agent registry
│   ├── compaction.py         # Context compaction
│   ├── context_epoch.py      # Epoch snapshots
│   ├── capability_calibrator.py  # Calibration tracking
│   └── domains/              # Domain-specific agents (5 domains)
├── storage/                  # Persistence layer (5 modules)
│   ├── session_store.py      # SQLite sessions
│   ├── blackboard.py         # Shared state
│   ├── decision_log.py       # Audit trail
│   └── artifact_store.py     # File artifacts
├── ui/                       # CustomTkinter desktop app (12 modules)
├── models/                   # Model adapters
├── roles/                    # Role definitions and prompts
├── schemas/                  # Data contracts
└── tests/                    # 1267 tests (48 files)
```

## Key Features

- **Zero external dependencies** — pure Python stdlib (sqlite3, threading, json, uuid)
- **1267 passing tests** — comprehensive unit and integration coverage
- **Dynamic role assignment** — roles adapt based on model capabilities and task requirements
- **Multi-model coordination** — pool management with health tracking and failover
- **Structured workflow** — complexity-driven mode selection (direct → expanded)
- **Budget control** — multi-dimensional tracking (tokens, calls, debates, retries)
- **Event-driven architecture** — pub/sub event bus with history replay
- **Streaming support** — SSE-style output with multi-consumer broadcast
- **Context management** — token-aware window with importance-based trimming
- **Permission matrix** — role-based access control with audit logging
- **Vector memory** — TF-IDF similarity search for knowledge retrieval

## License

MIT
