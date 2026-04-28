# BabyClaw

BabyClaw is a local coordinator-driven agentic system built around a routed
planner-executor-reviewer workflow. It uses a local Ollama model to plan,
generate responses, review execution results, and decide what long-term memory
should be stored.

The system is designed so the LLM does not directly mutate the workspace.
Instead, the LLM proposes structured plans, the infrastructure validates those
plans, the executor runs only registered tools, and a reviewer checks the result
before the workflow is considered successful.

## Requirements

- Python 3.11+
- Ollama installed and running
- A local Ollama model pulled, for example:

```bash
ollama pull qwen2.5:3b
```

Python dependencies:

```bash
pip install -r requirements.txt
```

## Running BabyClaw

Run the command-line interface:

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

## Basic CLI Commands

Inside the CLI, you can type normal tasks such as:

```text
Explain recursion simply
Create notes.txt with hello
Read README.md
Summarise README.md
Create a simple hello world Python file called hello.py
```

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

The workspace path controls where file tools can read and write. File operations
are sandboxed inside the configured workspace.

## How The System Works

The main runtime pipeline is:

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

### 1. Coordinator

The `Coordinator` is the main orchestrator. It receives the user's task and
controls the whole workflow.

It is responsible for:

- storing the user message,
- running the router,
- building planner input,
- retrying failed plans,
- running the executor,
- asking the user for permission before workspace mutations,
- sending execution results to the reviewer,
- rolling back rejected or failed mutations,
- triggering replanning after reviewer rejection,
- storing long-term memory after successful workflows,
- returning the final user-facing result.

Relevant file:

```text
src/Agents/Coordinator.py
```

### 2. RouteAgent

The `RouteAgent` classifies the current user task. It does not answer the user
and does not choose tools directly.

It classifies tasks into:

- `direct_response`
- `memory_question`
- `contextual_followup`
- `workspace_read`
- `workspace_summarise`
- `workspace_mutation`

Relevant files:

```text
src/Agents/Routing/RouteAgent.py
src/Agents/Routing/WorkflowPolicy.py
```

### 3. WorkflowPolicy

`WorkflowPolicyRegistry` converts the route classification into a strict
workflow contract.

The policy decides:

- which tools are allowed,
- whether workspace access is allowed,
- whether mutations are allowed,
- whether recent messages are included,
- which memory mode is used.

This keeps the LLM scoped. For example, a normal chat answer should not receive
file mutation tools.

### 4. PlannerAgent

The `PlannerAgent` converts the current task into a small JSON plan using only
the tools selected by the coordinator.

The planner is instructed to:

- use the fewest useful steps,
- preserve the current user task,
- avoid inventing files, paths, tools, or arguments,
- use direct arguments when known,
- use `*_step` arguments only when a value comes from an earlier step,
- avoid treating every use of the word "write" as a file operation.

Relevant files:

```text
src/Agents/Planning/PlannerAgent.py
src/Agents/Planning/PlannerPrompt.py
```

Example plan shape:

```json
{
  "goal": "Create hello.py with a simple hello world function.",
  "steps": [
    {
      "id": 1,
      "tool": "create_file",
      "args": {
        "path": "hello.py",
        "content": "def hello_world():\n    print('Hello world')\n"
      }
    }
  ],
  "planning_rationale": "The user asked to create a file with known content."
}
```

### 5. PlanCompiler

The `PlanCompiler` validates and normalises the planner output before execution.

It rejects common bad plans, including:

- unknown tools,
- unknown arguments,
- missing required arguments,
- invalid `*_step` references,
- future step references,
- fake step references inside strings,
- unsafe paths,
- absolute paths,
- path traversal attempts,
- workspace mutations targeting the workspace root,
- workspace mutation tasks that do not include a real mutation tool.

Relevant file:

```text
src/Agents/Planning/PlanCompiler.py
```

### 6. ExecutorAgent

The `ExecutorAgent` runs compiled plan steps through the registered tools.

It tracks:

- remaining steps,
- completed step results,
- step status,
- execution trace,
- approved actions,
- rollback snapshots,
- route scope.

Before executing a step, it checks that the tool is allowed by the current route
and that mutations are permitted.

Relevant file:

```text
src/Agents/ExecutorAgent.py
```

### 7. Tools

Planner-facing tool descriptions live in:

```text
src/tools/tool_description.py
```

Runtime tool registration lives in:

```text
src/tools/tool_registry.py
```

File tools live in:

```text
src/tools/file_tools.py
```

LLM-backed tools live in:

```text
src/tools/llm_tools.py
```

Available tools include:

- `direct_response`
- `generate_content`
- `read_file`
- `summarise_txt`
- `list_dir`
- `list_tree`
- `find_file`
- `find_file_recursive`
- `search_text`
- `create_file`
- `write_file`
- `append_file`
- `delete_file`
- `replace_text`
- `create_dir`
- `delete_dir`
- `move_path`
- `copy_path`

### 8. Permission And Rollback

Workspace mutation tools require user permission. Examples include:

- `create_file`
- `write_file`
- `append_file`
- `delete_file`
- `replace_text`
- `create_dir`
- `delete_dir`
- `move_path`
- `copy_path`

Before running those tools, the coordinator returns a permission request showing
the pending tools and resolved arguments.

If the user approves, execution continues. If the user rejects, the workflow is
cancelled.

For mutating tools, rollback snapshots are taken before execution. If execution
fails or the reviewer rejects the result, the coordinator uses those snapshots
to undo side effects where possible.

### 9. ReviewerAgent

The `ReviewerAgent` checks whether the executed result satisfies the current
user task.

It performs deterministic checks for objective workflow facts:

- executed tools must be inside `route_scope.allowed_tools`,
- mutation tools must not run when mutations are not allowed,
- workspace mutation tasks must use at least one real mutation tool,
- workspace mutation tasks must show either path changes or a content-changing
  tool such as `write_file`, `append_file`, or `replace_text`.

It also asks the LLM reviewer to judge semantic sufficiency. For example, if the
user asks for a game but the created file only contains hello-world code, the
reviewer should reject it because the artifact does not meaningfully satisfy the
requested outcome.

For generated/written files, review evidence includes:

- execution trace,
- resolved arguments,
- written content,
- final saved file content when available,
- workspace before/after,
- workspace diff,
- route scope.

Relevant files:

```text
src/Agents/Reviewing/ReviewerAgent.py
src/Agents/Reviewing/ReviewPrompt.py
```

### 10. MemoryAgent

The `MemoryAgent` has two responsibilities.

First, it stores all workflow messages in SQLite so the system has a trace of
what happened.

Second, after a successful workflow, it asks the LLM whether the user's latest
message contains stable long-term memory. It stores only explicit user facts or
preferences, not temporary task details.

Examples of memory-worthy statements:

```text
My name is Tristin.
I prefer short explanations.
My friend Jake is helping me with this project.
```

Examples of things it should not store:

```text
Create hello.txt
Read README.md
Use write_file
The planner created a file
```

Relevant files:

```text
src/Agents/MemoryAgent.py
src/Memory/MessageRepository.py
src/Memory/VectorRepository.py
src/Memory/sql_database.py
```

SQLite messages are stored in:

```text
Memory/memory.db
```

Vector memories are stored in:

```text
Memory/chroma_db
```

## Project Structure

```text
src/
  Main.py                         CLI entry point
  gui.py                          Flask web UI
  OllamaClient.py                 Ollama wrapper
  message.py                      Shared message DTO

  Agents/
    BaseAgent.py                  Shared base agent helper
    Coordinator.py                Main workflow orchestrator
    ExecutorAgent.py              Tool executor
    MemoryAgent.py                Message and long-term memory manager

    Routing/
      RouteAgent.py               Classifies user tasks
      WorkflowPolicy.py           Converts task type into allowed scope/tools

    Planning/
      PlannerAgent.py             Builds structured tool plans
      PlannerPrompt.py            Planner system prompt
      PlanCompiler.py             Validates and normalises plans

    Reviewing/
      ReviewerAgent.py            Reviews execution result
      ReviewPrompt.py             Reviewer system prompt

  Memory/
    sql_database.py               SQLite connection/init
    MessageRepository.py          SQL message storage
    VectorRepository.py           Chroma vector memory storage

  tools/
    tool_description.py           Planner-facing tool definitions
    tool_registry.py              Runtime tool registry
    file_tools.py                 Workspace file operations
    llm_tools.py                  LLM-backed tools
    utils.py                      Workspace sandbox config

tests/
  PlannerTest.py
  ReviewerTest.py
  PipelineTest.py
  ExecutorTest.py
  MemoryAgentTest.py
```

## Testing

The test filenames do not follow pytest's default `test_*.py` pattern, so run
them explicitly:

```bash
pytest -q tests/PlannerTest.py tests/ReviewerTest.py tests/PipelineTest.py tests/ExecutorTest.py tests/MemoryAgentTest.py
```

The tests cover:

- planner/compiler validation,
- unsafe path rejection,
- fake step reference rejection,
- executor behavior,
- reviewer acceptance/rejection,
- semantic review evidence,
- rollback and replanning,
- memory behavior.

## Design Goal

BabyClaw is not designed to make hallucinations impossible. Instead, it reduces
risk by wrapping the LLM in deterministic infrastructure:

- route-scoped tool access,
- structured planning,
- plan compilation,
- workspace sandboxing,
- mutation permission,
- runtime route enforcement,
- reviewer checks,
- rollback,
- replanning,
- controlled memory storage.

The LLM still matters. A stronger model can produce better plans and better
content. The architecture makes those plans safer to execute and easier to
reject when they do not satisfy the user's task.
