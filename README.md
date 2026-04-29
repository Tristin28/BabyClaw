# BabyClaw

BabyClaw is a local coordinator-driven agentic system built around a routed
Planner → Executor → Reviewer workflow.

It uses Ollama as the local LLM runtime. The LLM is used for planning, response
generation, reviewing results, and deciding whether useful long-term memory
should be stored.

The main design goal is to keep the LLM controlled. The LLM does not directly
mutate the workspace. Instead, it proposes structured plans, while the system
validates, executes, reviews, and stores results through deterministic Python
code.

---

## Requirements

- Python 3.11+
- Ollama installed and running
- A local model pulled through Ollama
- Python dependencies installed

Install dependencies:

```bash
pip install -r requirements.txt
````

Pull a model:

```bash
ollama pull qwen2.5:3b
```

---

## Running BabyClaw

Run the CLI:

```bash
python -m src.Main
```

Run the web interface:

```bash
python -m src.gui
```

Then open:

```text
http://127.0.0.1:5000
```

---

## CLI Commands

Inside the CLI, the user can type normal natural-language tasks.

Utility commands include:

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

The workspace path controls where file tools can read and write. File operations
are sandboxed inside the configured workspace.

---

## Main System Structure

The main workflow is:

```text
User task
  -> Coordinator
  -> RouteAgent
  -> WorkflowPolicy
  -> PlannerAgent
  -> PlanCompiler
  -> ExecutorAgent
  -> ReviewerAgent
  -> MemoryAgent
  -> Final response
```

Each part has a focused responsibility.

---

## Coordinator

The `Coordinator` is the main orchestrator.

It receives the user task, controls the workflow, routes messages between agents,
handles permission requests, triggers rollback when needed, and returns the final
response to the user.

Relevant file:

```text
src/Agents/Coordinator.py
```

---

## RouteAgent

The `RouteAgent` classifies the user task.

It decides whether the task is a direct response, memory question, contextual
follow-up, workspace read, workspace summary, or workspace mutation.

Relevant files:

```text
src/Agents/Routing/RouteAgent.py
src/Agents/Routing/WorkflowPolicy.py
```

---

## WorkflowPolicy

The `WorkflowPolicyRegistry` converts the route classification into a strict
workflow scope.

It decides:

* which tools are allowed,
* whether workspace access is allowed,
* whether mutation is allowed,
* whether recent messages should be included,
* which memory mode should be used.

This prevents the LLM from receiving unnecessary or unsafe tools.

Relevant file:

```text
src/Agents/Routing/WorkflowPolicy.py
```

---

## PlannerAgent

The `PlannerAgent` converts the user task into a small structured plan.

The planner can only use tools allowed by the active workflow policy. Its output
is not executed directly. It must first pass through the `PlanCompiler`.

Relevant files:

```text
src/Agents/Planning/PlannerAgent.py
src/Agents/Planning/PlannerPrompt.py
```

---

## PlanCompiler

The `PlanCompiler` validates and normalises the planner output before execution.

It rejects invalid plans, unsafe paths, unknown tools, missing arguments, invalid
step references, and unsafe workspace mutations.

Relevant file:

```text
src/Agents/Planning/PlanCompiler.py
```

---

## ExecutorAgent

The `ExecutorAgent` runs the compiled plan using registered tools.

It checks that every tool is allowed by the current route scope and that mutation
tools only run when mutation is permitted and approved.

Relevant file:

```text
src/Agents/ExecutorAgent.py
```

---

## Tools

Tools are split into descriptions and runtime implementations.

Planner-facing tool descriptions:

```text
src/tools/tool_description.py
```

Runtime tool registry:

```text
src/tools/tool_registry.py
```

File tools:

```text
src/tools/file_tools.py
```

LLM-backed tools:

```text
src/tools/llm_tools.py
```

Available tools include file reading, file writing, directory operations, text
searching, direct responses, and LLM-backed content generation.

---

## Permission and Rollback

Workspace mutation tools require user permission before execution.

Mutation tools include actions such as creating, writing, deleting, moving, or
copying files and directories.

Before a mutation runs, the coordinator shows the pending action and waits for
approval.

If execution fails or the reviewer rejects the result, BabyClaw attempts to roll
back workspace changes where possible.

---

## ReviewerAgent

The `ReviewerAgent` checks whether the execution result satisfies the original
user task.

It performs deterministic checks over workflow rules and also uses the LLM to
judge whether the result is semantically good enough.

Relevant files:

```text
src/Agents/Reviewing/ReviewerAgent.py
src/Agents/Reviewing/ReviewPrompt.py
```

---

## MemoryAgent

The `MemoryAgent` stores workflow messages and selected long-term memories.

It stores message history in SQLite so the system has a trace of what happened.

It stores long-term memory in a vector database only when the user gives stable
and useful information that may help future conversations.

Relevant files:

```text
src/Agents/MemoryAgent.py
src/Memory/MessageRepository.py
src/Memory/VectorRepository.py
src/Memory/sql_database.py
```

SQLite message history:

```text
Memory/memory.db
```

Vector memory:

```text
Memory/chroma_db
```

---

## Message Instances in the Memory Tab

A message instance is one stored workflow message.

It represents one communication between the user, coordinator, or one of the
agents.

A message instance usually contains:

```text
conversation_id: which conversation the message belongs to
step_index: where the message happened in the workflow
sender: who created the message
receiver: who received the message
target_agent: which agent should handle it next, if any
message_type: what kind of message it is
status: whether it completed, failed, or is waiting
response: the main stored content
visibility: whether it is internal or user-facing
timestamp: when it was created
```

The memory tab may look technical because it stores the system’s internal
workflow trace. It is not random data. It shows how a task moved through
BabyClaw.

---

## Project Structure

```text
src/
  Main.py                         CLI entry point
  gui.py                          Flask web UI
  OllamaClient.py                 Ollama communication wrapper
  message.py                      Shared message object

  Agents/
    Coordinator.py                Main workflow orchestrator
    ExecutorAgent.py              Tool executor
    MemoryAgent.py                Message and memory manager

    Routing/
      RouteAgent.py               Classifies user tasks
      WorkflowPolicy.py           Defines allowed workflow scope

    Planning/
      PlannerAgent.py             Builds structured plans
      PlannerPrompt.py            Planner prompt
      PlanCompiler.py             Validates plans before execution

    Reviewing/
      ReviewerAgent.py            Reviews execution results
      ReviewPrompt.py             Reviewer prompt

  Memory/
    sql_database.py               SQLite setup
    MessageRepository.py          Stores workflow messages
    VectorRepository.py           Stores long-term vector memories

  tools/
    tool_description.py           Tool descriptions for the planner
    tool_registry.py              Runtime tool registry
    file_tools.py                 Workspace file operations
    llm_tools.py                  LLM-backed tools
    utils.py                      Workspace sandbox helpers

tests/
  PlannerTest.py
  ReviewerTest.py
  PipelineTest.py
  ExecutorTest.py
  MemoryAgentTest.py
```

---

## Testing

Run tests explicitly:

```bash
pytest -q tests/PlannerTest.py tests/ReviewerTest.py tests/PipelineTest.py tests/ExecutorTest.py tests/MemoryAgentTest.py
```

The tests cover planning, compiling, execution, reviewing, rollback, replanning,
and memory behaviour.

---

## Design Goal

BabyClaw is designed to make agent workflows more controlled and inspectable.

It reduces risk through:

* route-scoped tool access,
* structured planning,
* plan validation,
* workspace sandboxing,
* mutation permission,
* reviewer checks,
* rollback support,
* controlled memory storage.

The LLM still helps with reasoning, but deterministic code controls how the
workflow actually runs.

```
```
