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

- **Direct chat responses**; explanations, advice, drafting text in chat,
  answering general questions ("what is reinforcement learning?").
- **Memory questions**; questions about durable user facts/preferences the
  system has stored ("what is my name?").
- **Contextual follow-ups**; requests that depend on previous conversation
  ("make it shorter", "continue", "save that to a file").
- **Workspace read** ; read, list, find, search files/folders inside the
  configured workspace.
- **Workspace summarise** ; summarise or describe an existing file.
- **Workspace mutation** ; create, write, append, replace, delete, move,
  copy files or directories.

All file changes are sandboxed inside the configured workspace and require
explicit user approval at the CLI/GUI before they run.

## How The System Works
<img width="822" height="722" alt="BabyClaw drawio" src="https://github.com/user-attachments/assets/2a4b337a-d23b-4252-b933-614bd6d45f2f" />

## Requirements

- Python 3.11+
- Ollama installed and running
- A local Ollama model
- Python dependencies from `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull a local model - Example:

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

## Design Goals

BabyClaw exists to keep a local agent workflow inspectable and controlled.
The LLM provides reasoning. Deterministic code decides what the LLM sees,
what tools it can use, how its plans are validated, how its actions are
executed, and whether the result is acceptable. Every mutation is
permissioned, snapshotted, verified, and reviewable.
