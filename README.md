# BabyClaw

BabyClaw is a local, coordinator-driven agent system. It uses Ollama as the
local LLM runtime. The LLM is responsible for understanding what the user
means: it classifies the task, plans, generates text, and reviews semantic
quality. Everything that touches the filesystem, the permissions model, the
workflow scope, validation, verification, and rollback is handled by
deterministic Python.

The LLM never writes to disk directly. File changes go through structured
plans, plan validation, permission prompts, sandboxed tools, post-execution
verification, and reviewer acceptance before anything is considered done.

This project is intended for local exploration and assignment-style work. It
is not production software.

## What BabyClaw Can Do

The router classifies each request into one of six task types. The Coordinator
turns the task type into a workflow contract (allowed tools, memory scope,
permissions). Supported task types:

- **Direct chat responses** — explanations, advice, drafting text in chat,
  answering general questions ("what is reinforcement learning?").
- **Memory questions** — questions about durable user facts/preferences the
  system has stored ("what is my name?").
- **Contextual follow-ups** — requests that depend on previous conversation
  ("make it shorter", "continue", "save that to a file").
- **Workspace read** — read, list, find, search files/folders inside the
  configured workspace.
- **Workspace summarise** — summarise or describe an existing file.
- **Workspace mutation** — create, write, append, replace, delete, move,
  copy files or directories.

All file changes are sandboxed inside the configured workspace and require
explicit user approval at the CLI/GUI before they run.

## How The System Works

```text
User task
  -> Coordinator
  -> RouteAgent
  -> WorkflowPolicyRegistry
  -> MemoryRoutingPolicy
  -> ContextResolver / ActiveContext
  -> PlannerAgent
  -> PlanCompiler
  -> ExecutorAgent
  -> ExecutionVerifier
  -> ReviewerAgent
  -> MemoryAgent
  -> Final response / Replan / Rollback
```

What each stage does:

- **Coordinator** — orchestrates the workflow, builds the layered planner
  input, runs the execution loop, handles permission prompts, replans, and
  rolls back on failure.
- **RouteAgent** — an LLM call that classifies the task into one of the six
  task types. The only deterministic override left is a chat guard: if the
  user says "in chat", the route cannot become a file mutation.
- **WorkflowPolicyRegistry** — frozen per-task-type contract that defines
  `allowed_tools`, `allow_mutations`, `memory_mode`, and whether the
  workspace tree is exposed.
- **MemoryRoutingPolicy** — decides whether short-term recent messages and
  long-term vector memory should be visible for this task. Standalone chat
  questions do not pull memory; contextual follow-ups do.
- **ActiveContext** — small in-memory record of the last assistant response,
  last generated content, last viewed/modified/created file, and active
  file.
- **ContextResolver** — deterministically maps phrases like "it", "the
  previous answer", "the file" to concrete entries in `ActiveContext` so
  the planner sees real values instead of pronouns.
- **PlannerAgent** — LLM call that produces a structured plan (`goal`,
  `steps`, `planning_rationale`) using only the tools the route allows. A
  JSON schema with a tool-name enum is enforced at the Ollama layer.
- **PlanCompiler** — pure Python validator. Rejects unknown tools, fake
  step references, missing required args, unsafe paths, paths outside the
  user's requested filename, plans that claim a mutation route but contain
  no mutation tool, and so on. The planner cannot bypass it.
- **ExecutorAgent** — runs only compiled plans, and only via the tool
  registry. Re-checks `allowed_tools` and `allow_mutations` per step,
  takes a rollback snapshot before every mutation, and pauses the loop
  whenever a step requires user permission.
- **ExecutionVerifier** — independent of the reviewer; after execution it
  reads the live filesystem and checks whether each mutating step produced
  the file state its resolved args claim.
- **ReviewerAgent** — semantic check. Compares the result against the
  current user task using deterministic structural checks (route scope,
  unrequested mutations, literal-text presence) plus an LLM judgement for
  topic relevance.
- **MemoryAgent** — stores every workflow message to SQLite. After an
  accepted result, decides via an LLM call whether anything in the user
  task should become a durable user fact or preference in the vector
  store.

## Context Handling

The current user task is always the primary goal. Every other source of
context is supporting evidence and must not replace the task. The planner
prompt receives context in clearly labelled layers:

1. `PRIMARY GOAL: CURRENT USER TASK`
2. `RESOLVED CONTEXTUAL REFERENCES` (`ContextResolver` mapping pronouns to
   real state)
3. `RECENT CONVERSATION CONTEXT` (last user/assistant turns; used for
   contextual follow-ups)
4. `RELEVANT LONG-TERM MEMORY` (vector store, only when the memory policy
   says it is useful)
5. `WORKSPACE CONTEXT` (recursive file listing, included for any
   workspace-aware route and for `contextual_followup` turns where an
   active file is recorded)
6. `FILENAME RESOLUTION HINTS` (Coordinator-resolved name → path mapping)
7. `ROUTE / POLICY` (authoritative scope)
8. `ALLOWED TOOLS`
9. `HARD USER CONSTRAINTS` (locked paths)

Filename hints let users refer to a file without typing the extension:
"open Coordinator" is resolved to `Coordinator.py` if that is the only
match. On mutation routes ambiguous matches are hidden from the planner so
it cannot accidentally write to the wrong file. The planner prompt is
instructed to ask for clarification when multiple matches are surfaced.

If context conflicts with the current user task, the current user task
wins.

## Safety Model

The design rule is: **the LLM decides what the user means; deterministic
Python decides whether the proposed action is safe, valid, allowed,
executable, and verified.**

Concrete safeguards layered behind the LLM:

- **Route-scoped tool access** — `WorkflowPolicyRegistry` returns an
  immutable per-task-type contract. Planner can only emit tools in
  `allowed_tools`; executor refuses anything outside.
- **`allow_mutations` flag** — mutation tools refuse to run on read or
  summarise routes even if a malformed plan slips past the compiler.
- **Structured plan validation** — `PlanCompiler` rejects bad schemas,
  fake step references, unknown tools, missing args, future
  dependencies, and self-dependencies.
- **Workspace sandbox** — every path is resolved through
  `WorkspaceConfig.resolve_workspace_path`. Absolute paths, `..`
  traversal, and symlink escapes are blocked.
- **Symlink hardening** — `Path.resolve()` plus `os.path.realpath()`
  cross-check, plus an explicit parent-chain symlink scan. A symlink
  inside the workspace pointing outside cannot be read, written,
  deleted, moved, or copied.
- **Permission prompts** — every mutation tool is marked
  `requires_permission=True`. The execution loop pauses and asks the
  user before running.
- **Rollback snapshots** — every mutation tool registers a snapshot
  before it runs. On failure, verifier rejection, or reviewer
  rejection, the Coordinator replays snapshots in reverse.
- **Execution verification** — `ExecutionVerifier` reads the live
  filesystem after execution and compares it against the resolved args.
- **Reviewer checks** — deterministic checks for route scope,
  unrequested mutations, and literal text presence, plus an LLM call
  for topic relevance.
- **Loop and retry limits** — `MAX_WORKFLOW_ITERATIONS=3`,
  `MAX_EXECUTION_LOOP_ITERATIONS=25`, `MAX_REPEATED_EXECUTION_STATES=2`.

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

By default the Ollama client uses `qwen2.5:3b`. To use a different local
model, pass it when constructing `OllamaClient` in `src/app/Main.py` or
`src/app/gui.py`:

```python
llm_client = OllamaClient(model="llama3.1:8b")
```

## CLI Commands

Inside the CLI you type natural-language tasks directly. Utility commands:

```text
help
exit
set workspace <path>
show plans
show plans <number>
show llm
show llm <number>
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

The workspace path controls where file tools can read and write. All
workspace operations are sandboxed inside the configured workspace.

## Project Structure

```text
src/
  app/
    Main.py                    CLI entry point
    gui.py                     Flask web UI
    templates/index.html       GUI template

  core/
    context/
      ActiveContext.py         Session state (last response, active file, ...)
      ContextResolver.py       Pronoun -> concrete state resolver
    workflow/
      Coordinator.py           Workflow orchestrator
      ExecutionVerifier.py     Deterministic post-execution checks
    message.py                 Workflow Message DTO

  agents/
    BaseAgent.py               Shared agent base class

    routing/
      RouteAgent.py            LLM task classifier
      WorkflowPolicy.py        Per-task-type tool/scope contract
      MemoryRoutingPolicy.py   Per-task-type memory scope

    planning/
      PlannerAgent.py          Structured plan builder
      PlannerPrompt.py         Planner system prompt
      PlanCompiler.py          Plan validator and normaliser

    execution/
      ExecutorAgent.py         Runs compiled plans via tool registry

    reviewing/
      ReviewerAgent.py         Semantic + structural reviewer
      ReviewPrompt.py          Reviewer system prompt

    memory/
      MemoryAgent.py           Message log + durable memory writer

  memory/
    sql_database.py            SQLite setup
    MessageRepository.py       SQLite message persistence
    VectorRepository.py        Chroma vector memory

  llm/
    OllamaClient.py            Ollama API wrapper + raw call logging

  tools/
    tool_description.py        Planner-facing tool descriptions
    tool_registry.py           Runtime tool registry (func, input_map,
                               permission flag, rollback hooks)
    file_tools.py              Sandboxed workspace file operations
    llm_tools.py               direct_response, generate_content,
                               summarise_txt
    utils.py                   WorkspaceConfig path resolver

  config/
    workspace_config.py        Workspace path persistence

config/
  workspace_config.json        Saved workspace path

tests/                          Unit and integration tests (see below)

Memory/                         Runtime SQLite + vector + LLM call log
workspace/                      Default workspace folder
```

## Important Runtime Paths

```text
config/workspace_config.json   Saved workspace path
workspace/                     Default workspace folder
Memory/memory.db               SQLite workflow message history
Memory/chroma_db/              Chroma vector store
Memory/llm_calls.jsonl         Raw LLM call log
```

For a clean local run, remove generated files under `Memory/` and reset the
workspace path in `config/workspace_config.json`.

## Memory

BabyClaw keeps two kinds of memory:

- **SQLite message history** at `Memory/memory.db` — every agent message
  is logged for inspection and debugging.
- **Vector long-term memory** at `Memory/chroma_db` — durable user facts
  and preferences. The LLM is asked whether anything in the user task is
  memory-worthy, the candidate memories are validated by Python rules,
  and only then are they written. This only happens after a workflow
  result is accepted by the reviewer.

Immediate contextual references are handled separately. `ActiveContext`
lives in memory for the running session and is updated only after a
successful workflow, so a rejected/rolled-back attempt cannot poison
later turns.

## Design Goals

BabyClaw exists to keep a local agent workflow inspectable and controlled.
The LLM provides reasoning. Deterministic code decides what the LLM sees,
what tools it can use, how its plans are validated, how its actions are
executed, and whether the result is acceptable. Every mutation is
permissioned, snapshotted, verified, and reviewable.