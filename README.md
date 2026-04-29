# BabyClaw

BabyClaw is a local coordinator-driven agent system built around a routed
Planner -> Executor -> Reviewer workflow.

It uses Ollama as the local LLM runtime. The LLM helps with routing, planning,
response generation, review, and long-term memory selection, but it does not
directly mutate the workspace. File and directory changes go through structured
plans, validation, permission checks, deterministic Python tools, review, and
rollback support.

## Requirements

- Python 3.11+
- Ollama installed and running
- A local Ollama model
- Python dependencies from `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull a local model:

```bash
ollama pull qwen2.5:3b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Running

Run the CLI:

```bash
python -m src.app.Main
```

Run the web interface:

```bash
python -m src.app.gui
```

The GUI runs at:

```text
http://127.0.0.1:5000
```

## CLI Commands

Inside the CLI, you can type natural-language tasks directly.

Utility commands:

```text
help
exit
set workspace <path>
show plans
show plans <number>
show memory
show memory <number>
debug paths
debug recent
debug recent <number>
debug context
debug context <number>
debug on
debug off
```

The workspace path controls where file tools can read and write. Workspace file
operations are sandboxed inside the configured workspace.

## How The System Works

The core workflow is:

```text
User task
  -> Coordinator
  -> RouteAgent
  -> WorkflowPolicyRegistry
  -> PlannerAgent
  -> PlanCompiler
  -> ExecutorAgent
  -> ExecutionVerifier
  -> ReviewerAgent
  -> MemoryAgent
  -> Final response
```

Responsibilities:

- `Coordinator` orchestrates the workflow and moves messages between agents.
- `RouteAgent` classifies the task type.
- `WorkflowPolicyRegistry` converts the route into allowed tools, workspace
  access, mutation rights, and memory scope.
- `MemoryRoutingPolicy` decides whether short-term or long-term memory should be
  visible for the current task.
- `PlannerAgent` produces a structured plan using only route-approved tools.
- `PlanCompiler` validates and normalises the plan before anything runs.
- `ExecutorAgent` executes approved plan steps through the registered tools.
- `ExecutionVerifier` checks deterministic execution constraints.
- `ReviewerAgent` reviews whether the result satisfies the original task.
- `MemoryAgent` stores workflow messages and selected durable memories.

## Permissions And Rollback

Workspace mutation tools require user permission before execution.

Mutation tools include creating, writing, appending, deleting, replacing, moving,
or copying files and directories.

Before a mutation runs, BabyClaw shows the pending action and waits for approval.
If execution fails, the verifier rejects the result, or the reviewer rejects the
result, BabyClaw attempts to roll back workspace changes where possible.

## Memory

BabyClaw stores two kinds of runtime memory:

- SQLite message history in `Memory/memory.db`
- Vector long-term memory in `Memory/chroma_db`

The GUI memory tab exposes the internal workflow trace. These message records are
technical by design: they show which agent produced which message, at which
workflow step, and with what status.

## Software Architecture

The source tree is split by responsibility:

```text
src/
  app/
    Main.py                    CLI entry point
    gui.py                     Flask web UI
    templates/
      index.html               GUI template

  core/
    message.py                 Shared workflow message object
    workflow/
      Coordinator.py           Main workflow orchestrator
      ExecutionVerifier.py     Deterministic execution verification

  agents/
    BaseAgent.py               Shared agent base class

    planning/
      PlannerAgent.py          Builds structured plans
      PlannerPrompt.py         Planner system prompt
      PlanCompiler.py          Validates and normalises plans

    routing/
      RouteAgent.py            Classifies user tasks
      WorkflowPolicy.py        Defines route-scoped tool access
      MemoryRoutingPolicy.py   Defines task-scoped memory access

    execution/
      ExecutorAgent.py         Executes compiled plan steps

    reviewing/
      ReviewerAgent.py         Reviews execution results
      ReviewPrompt.py          Reviewer system prompt

    memory/
      MemoryAgent.py           Stores messages and long-term memories

  memory/
    sql_database.py            SQLite setup
    MessageRepository.py       Message persistence
    VectorRepository.py        Vector-memory persistence

  llm/
    OllamaClient.py            Ollama API wrapper and LLM call logging

  tools/
    tool_description.py        Planner-facing tool descriptions
    tool_registry.py           Runtime tool registry
    file_tools.py              Workspace file operations
    llm_tools.py               LLM-backed tools
    utils.py                   Workspace sandbox helpers

  config/
    workspace_config.py        Workspace path persistence

tests/
  PlannerTest.py
  ReviewerTest.py
  PipelineTest.py
  ExecutorTest.py
  MemoryAgentTest.py
  RouteAgentTest.py
  ExecutionVerifierTest.py
  OllamaClientTest.py
  test_architecture_guards.py
```

## Important Runtime Paths

```text
config/workspace_config.json   Saved workspace path
workspace/                     Default workspace folder
Memory/memory.db               SQLite workflow message history
Memory/chroma_db/              Chroma vector store
Memory/llm_calls.jsonl         Raw LLM call log
```

Runtime data can grow over time. If you want a clean local run, remove generated
files under `Memory/` and reset the workspace path in `config/workspace_config.json`.

## Testing

Install test tooling if it is not already available:

```bash
pip install pytest
```

Run all tests:

```bash
pytest -q tests
```

Run a focused subset:

```bash
pytest -q tests/PlannerTest.py tests/ReviewerTest.py tests/PipelineTest.py
```

The tests cover route scoping, planning, plan compilation, execution, review,
rollback, replanning, memory behaviour, and architecture guard cases.

## Design Goals

BabyClaw is designed to keep local agent workflows controlled and inspectable.

The main safeguards are:

- route-scoped tool access,
- structured planning,
- plan validation,
- workspace sandboxing,
- explicit permission for mutations,
- deterministic execution checks,
- reviewer checks,
- rollback support,
- controlled memory visibility and storage.

The LLM provides reasoning, but deterministic code decides what context it sees,
which tools it may use, how plans are validated, and how workspace effects are
applied.
