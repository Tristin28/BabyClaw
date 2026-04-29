from src.agents.execution.ExecutorAgent import ExecutorAgent
from src.core.workflow.ExecutionVerifier import ExecutionVerifier
from src.agents.reviewing.ReviewerAgent import ReviewerAgent
from src.core.workflow.Coordinator import Coordinator
from src.agents.routing.WorkflowPolicy import WorkflowPolicyRegistry
from src.core.message import Message
from src.tools.tool_registry import build_tool_registry
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.utils import WorkspaceConfig


class FakeMemory:
    def __init__(self):
        self.messages = []

    def store_message(self, message):
        self.messages.append(message)
        return message

    def get_relevant_memory(self, task: str, k: int):
        return ""

    def get_recent_conversation_messages(self, conversation_id: int, k: int = 5):
        return []

    def store_long_term_memory(self, user_task: str, conversation_id: int, step_index: int):
        return Message(
            conversation_id=conversation_id,
            step_index=step_index,
            sender="memory",
            receiver="coordinator",
            target_agent=None,
            message_type="memory_store",
            status="completed",
            response={"stored": False},
            visibility="internal"
        )


class FakePlanner:
    """
    First plan intentionally hallucinates an extra workspace mutation.
    Second plan is corrected after reviewer rejection.
    """

    def __init__(self):
        self.calls = []
        self.name = "planner"

    def run(self, planner_input: dict):
        self.calls.append(planner_input)

        if len(self.calls) == 1:
            response = {
                "goal": "Create work.txt with hello, but hallucinate an extra directory",
                "steps": [
                    {
                        "id": 1,
                        "tool": "create_file",
                        "args": {
                            "path": "work.txt",
                            "content": "hello"
                        },
                        "depends_on": []
                    },
                    {
                        "id": 2,
                        "tool": "create_dir",
                        "args": {
                            "path": "unrelated_dir"
                        },
                        "depends_on": []
                    }
                ],
                "planning_rationale": "Bad hallucinated plan."
            }
        else:
            response = {
                "goal": "Create work.txt with hello",
                "steps": [
                    {
                        "id": 1,
                        "tool": "create_file",
                        "args": {
                            "path": "work.txt",
                            "content": "hello"
                        },
                        "depends_on": []
                    }
                ],
                "planning_rationale": "Corrected plan after reviewer rejection."
            }

        return Message(
            conversation_id=planner_input["conversation_id"],
            step_index=planner_input["step_index"],
            sender="planner",
            receiver="coordinator",
            target_agent="executor",
            message_type="plan",
            status="completed",
            response=response,
            visibility="internal"
        )


class FakeReviewer:
    """
    Rejects the first execution because it contains create_dir on unrelated_dir.
    Accepts the second execution.
    """

    def __init__(self):
        self.calls = []

    def build_review_evidence(self, executor_response: dict, workspace_before=None, workspace_after=None) -> dict:
        clean_steps = []

        for step in executor_response.get("execution_trace", []):
            clean_step = {
                "id": step.get("id"),
                "tool": step.get("tool"),
                "status": step.get("status"),
                "args": step.get("args", {}),
            }

            if "resolved_args" in step:
                clean_step["resolved_args"] = step["resolved_args"]

            if "result" in step:
                clean_step["result"] = step["result"]

            if "error" in step:
                clean_step["error"] = step["error"]

            clean_steps.append(clean_step)

        return {
            "goal": executor_response.get("goal", ""),
            "steps": clean_steps,
            "workspace_before": workspace_before or executor_response.get("workspace_before", []),
            "workspace_after": workspace_after or executor_response.get("workspace_after", []),
        }

    def run(self, conversation_id: int, step_index: int, user_task: str,
            execution_response: dict, workspace_before=None, workspace_after=None, route=None):

        self.calls.append({
            "user_task": user_task,
            "execution_response": execution_response,
            "workspace_before": workspace_before or [],
            "workspace_after": workspace_after or []
        })

        trace = execution_response.get("execution_trace", [])

        for step in trace:
            args = step.get("resolved_args") or step.get("args", {})

            if step.get("tool") == "create_dir" and args.get("path") == "unrelated_dir":
                return Message(
                    conversation_id=conversation_id,
                    step_index=step_index,
                    sender="reviewer",
                    receiver="coordinator",
                    target_agent=None,
                    message_type="review_result",
                    status="completed",
                    response={
                        "accepted": False,
                        "review_summary": "Execution included an unrequested directory creation.",
                        "issues": ["create_dir on unrelated_dir was not requested by the current task."]
                    },
                    visibility="internal"
                )

        return Message(
            conversation_id=conversation_id,
            step_index=step_index,
            sender="reviewer",
            receiver="coordinator",
            target_agent=None,
            message_type="review_result",
            status="completed",
            response={
                "accepted": True,
                "review_summary": "The requested file was created correctly.",
                "issues": []
            },
            visibility="internal"
        )


class FakeLLM:
    def invoke_json(self, *args, **kwargs):
        return {"should_store": False, "memories": []}

    def invoke_text(self, *args, **kwargs):
        return "summary"


class FakeRouter:
    def __init__(self):
        self.name = "router"

    def run(self, conversation_id: int, step_index: int, user_task: str):
        return Message(
            conversation_id=conversation_id,
            step_index=step_index,
            sender="router",
            receiver="coordinator",
            target_agent=None,
            message_type="route",
            status="completed",
            response={
                "task_type": "workspace_mutation",
                "confidence": 1.0,
                "routing_reason": "Test task mutates the workspace."
            },
            visibility="internal"
        )


def build_test_system(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    workspace = WorkspaceConfig(root=str(workspace_root))
    llm = FakeLLM()

    tool_registry = build_tool_registry(
        llm_client=llm,
        workspace=workspace
    )

    planner = FakePlanner()
    executor = ExecutorAgent(tool_registry=tool_registry)
    reviewer = FakeReviewer()
    memory = FakeMemory()
    router = FakeRouter()

    coordinator = Coordinator(
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm,
        router=router
    )

    return coordinator, workspace, planner, reviewer, workspace_root


def approve_permission_if_needed(coordinator, message):
    """
    The real system asks permission before workspace mutations.
    This helper simulates the user saying yes.
    """
    if message.message_type != "permission_request":
        return message

    response = message.response

    return coordinator.continue_after_permission(
        conversation_id=1,
        user_task=response["user_task"],
        plan_response=response["plan_response"],
        execution_state=response["execution_state"],
        pending_runnable_steps=response["pending_runnable_steps"],
        step_index=response["next_step_index"],
        approved=True
    )


def test_reject_then_rollback_then_replan_then_succeed(tmp_path):
    coordinator, workspace, planner, reviewer, workspace_root = build_test_system(tmp_path)

    message = coordinator.start_workflow(
        conversation_id=1,
        user_task="create work.txt with hello"
    )

    # First plan requires permission because it mutates files.
    message = approve_permission_if_needed(coordinator, message)

    # First execution should be rejected because it created unrelated_dir.
    # Coordinator should rollback and replan.
    # Second plan also needs permission.
    assert message.message_type == "permission_request"
    second_execution_state = message.response["execution_state"]
    assert second_execution_state["approved_actions"] == set()
    assert second_execution_state["approved_step_ids"] == set()
    assert second_execution_state["step_results"] == {}
    assert second_execution_state["step_status"] == {}
    assert second_execution_state["execution_trace"] == []
    assert len(second_execution_state["remaining_steps"]) == 1
    assert second_execution_state["remaining_steps"][0]["args"]["path"] == "work.txt"

    message = approve_permission_if_needed(coordinator, message)

    assert message.status == "completed"
    assert message.message_type == "workflow_result"

    assert (workspace_root / "work.txt").exists()
    assert (workspace_root / "work.txt").read_text(encoding="utf-8") == "hello"

    # This confirms rollback removed the hallucinated extra mutation.
    assert not (workspace_root / "unrelated_dir").exists()

    # Planner was called twice: first hallucinated, second corrected.
    assert len(planner.calls) == 2

    first_planner_input = planner.calls[0]
    second_planner_input = planner.calls[1]

    # The original task must stay unchanged.
    assert first_planner_input["task"] == "create work.txt with hello"
    assert second_planner_input["task"] == "create work.txt with hello"

    # Rejection feedback must go into context, not into task.
    assert "REPLAN CONTEXT" in second_planner_input["context"]
    assert "create_dir on unrelated_dir" in second_planner_input["context"]
    assert "create_dir on unrelated_dir" not in second_planner_input["task"]

    # Reviewer saw both executions.
    assert len(reviewer.calls) == 2


def test_planner_path_traversal_fails_before_permission_or_execution(tmp_path):
    """
        Unsafe paths should fail during plan compilation before permission or execution.
    """
    from src.agents.planning.PlannerAgent import PlannerAgent

    class PathTraversalLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            return {
                "goal": "Read outside workspace",
                "steps": [
                    {
                        "id": 1,
                        "tool": "read_file",
                        "args": {"path": "../../etc/passwd"}
                    }
                ],
                "planning_rationale": "Bad hallucinated path."
            }

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    workspace = WorkspaceConfig(root=str(workspace_root))

    planner = PlannerAgent(
        llm_client=PathTraversalLLM(),
        workspace_config=workspace
    )

    planner_input = {
        "task": "read hello.txt",
        "context": "",
        "k_recent_messages": [],
        "tools": PLANNER_TOOL_DESCRIPTIONS,
        "conversation_id": 1,
        "step_index": 1,
        "workspace_contents": []
    }

    msg = planner.run(planner_input)

    assert msg.status == "failed"
    assert "outside the workspace" in msg.response["error"] or "unsafe" in msg.response["error"]


def test_replan_preserves_user_task_verbatim(tmp_path):
    coordinator, workspace, planner, reviewer, workspace_root = build_test_system(tmp_path)

    message = coordinator.start_workflow(
        conversation_id=1,
        user_task="create work.txt with hello"
    )

    message = approve_permission_if_needed(coordinator, message)
    message = approve_permission_if_needed(coordinator, message)

    assert len(planner.calls) == 2

    original_task = "create work.txt with hello"
    replan_input = planner.calls[1]

    assert replan_input["task"] == original_task

    # Using lower() so this test does not fail just because of capital letters.
    assert "previous attempt was rejected" in replan_input["context"].lower()

    # Reviewer feedback should be present in context.
    assert "create_dir on unrelated_dir was not requested" in replan_input["context"]

    # Reviewer feedback should NOT pollute the task field.
    assert "create_dir on unrelated_dir was not requested" not in replan_input["task"]


def test_planning_retry_feedback_preserves_explicit_requested_path(tmp_path):
    from src.agents.planning.PlannerAgent import PlannerAgent

    class PathRepairLLM:
        def __init__(self):
            self.calls = []

        def invoke_json(self, messages, stream=False, schema=None):
            self.calls.append(messages)

            if len(self.calls) == 1:
                return {
                    "goal": "Create hello world program",
                    "steps": [
                        {
                            "id": 1,
                            "tool": "create_file",
                            "args": {
                                "path": "hello_world.py",
                                "content": "print('hello world')"
                            }
                        }
                    ],
                    "planning_rationale": "Bad inferred filename."
                }

            return {
                "goal": "Create hello world program at requested path",
                "steps": [
                    {
                        "id": 1,
                        "tool": "create_dir",
                        "args": {"path": "src"}
                    },
                    {
                        "id": 2,
                        "tool": "create_file",
                        "args": {
                            "path": "src/main.py",
                            "content": "print('hello world')"
                        }
                    }
                ],
                "planning_rationale": "Use the explicit requested path."
            }

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceConfig(root=str(workspace_root))
    llm = PathRepairLLM()

    planner = PlannerAgent(
        llm_client=llm,
        workspace_config=workspace
    )

    memory = FakeMemory()
    tool_registry = build_tool_registry(
        llm_client=llm,
        workspace=workspace
    )

    coordinator = Coordinator(
        planner=planner,
        executor=ExecutorAgent(tool_registry=tool_registry),
        reviewer=FakeReviewer(),
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm,
        router=FakeRouter()
    )

    user_task = "Create src/main.py with a simple Python hello world program"
    route = WorkflowPolicyRegistry.build_route({
        "task_type": "workspace_mutation",
        "confidence": 1.0,
        "routing_reason": "test"
    })
    planner_input = coordinator.build_planner_input(
        user_task=user_task,
        route=route,
        conversation_id=1,
        step_index=2
    )

    assert planner_input["requested_paths"] == ["src/main.py"]
    assert planner_input["allowed_parent_dirs"] == ["src"]

    msg = coordinator.try_plan(planner_input=planner_input)

    assert msg.status == "completed"
    assert len(llm.calls) == 2

    retry_prompt = llm.calls[1][-1]["content"]
    assert "The user explicitly requested src/main.py" in retry_prompt
    assert "Your plan used hello_world.py" in retry_prompt
    assert "Regenerate the plan using exactly src/main.py" in retry_prompt
    assert "The only allowed mutation paths are src and src/main.py" in retry_prompt
    assert "Do not invent hello_world.py, code.py, main.py, or any other filename" in retry_prompt
    assert "create only 'src'" in retry_prompt

    steps = msg.response["steps"]
    assert steps[0]["tool"] == "create_dir"
    assert steps[0]["args"]["path"] == "src"
    assert steps[1]["tool"] == "create_file"
    assert steps[1]["args"]["path"] == "src/main.py"


def test_planning_retry_feedback_explains_fake_content_step_string(tmp_path):
    from src.agents.planning.PlannerAgent import PlannerAgent

    class FakeStepRepairLLM:
        def __init__(self):
            self.calls = []

        def invoke_json(self, messages, stream=False, schema=None):
            self.calls.append(messages)

            if len(self.calls) == 1:
                return {
                    "goal": "Create hello world program",
                    "steps": [
                        {
                            "id": 1,
                            "tool": "create_file",
                            "args": {
                                "path": "src/main.py",
                                "content": "content_step:1"
                            }
                        }
                    ],
                    "planning_rationale": "Bad fake step string."
                }

            return {
                "goal": "Create hello world program at requested path",
                "steps": [
                    {
                        "id": 1,
                        "tool": "generate_content",
                        "args": {
                            "prompt": "Write a simple Python hello world program. Output only Python code."
                        }
                    },
                    {
                        "id": 2,
                        "tool": "create_file",
                        "args": {
                            "path": "src/main.py",
                            "content_step": 1
                        }
                    }
                ],
                "planning_rationale": "Use proper content_step syntax."
            }

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceConfig(root=str(workspace_root))
    llm = FakeStepRepairLLM()

    planner = PlannerAgent(
        llm_client=llm,
        workspace_config=workspace
    )
    memory = FakeMemory()
    tool_registry = build_tool_registry(
        llm_client=llm,
        workspace=workspace
    )

    coordinator = Coordinator(
        planner=planner,
        executor=ExecutorAgent(tool_registry=tool_registry),
        reviewer=FakeReviewer(),
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm,
        router=FakeRouter()
    )

    route = WorkflowPolicyRegistry.build_route({
        "task_type": "workspace_mutation",
        "confidence": 1.0,
        "routing_reason": "test"
    })
    planner_input = coordinator.build_planner_input(
        user_task="Create src/main.py with a simple Python hello world program",
        route=route,
        conversation_id=1,
        step_index=2
    )

    msg = coordinator.try_plan(planner_input=planner_input)

    assert msg.status == "completed"
    assert len(llm.calls) == 2

    retry_prompt = llm.calls[1][-1]["content"]
    assert "You used {'content': 'content_step'}, which is invalid" in retry_prompt
    assert "Use {'content_step': 1} instead" in retry_prompt

    steps = msg.response["steps"]
    assert steps[0]["tool"] == "generate_content"
    assert steps[1]["args"]["path"] == "src/main.py"
    assert steps[1]["args"]["content_step"] == 1


def test_no_permission_request_until_explicit_path_plan_is_valid(tmp_path):
    from src.agents.planning.PlannerAgent import PlannerAgent

    class BadThenValidPathLLM:
        def __init__(self):
            self.json_calls = []

        def invoke_json(self, messages, stream=False, schema=None):
            self.json_calls.append(messages)

            if len(self.json_calls) == 1:
                return {
                    "goal": "Create hello world program",
                    "steps": [
                        {
                            "id": 1,
                            "tool": "create_file",
                            "args": {
                                "path": "code.py",
                                "content": "print('hello world')"
                            }
                        }
                    ],
                    "planning_rationale": "Bad inferred filename."
                }

            return {
                "goal": "Create hello world program at requested path",
                "steps": [
                    {
                        "id": 1,
                        "tool": "generate_content",
                        "args": {
                            "prompt": "Write a simple Python hello world program. Output only Python code."
                        }
                    },
                    {
                        "id": 2,
                        "tool": "create_file",
                        "args": {
                            "path": "src/main.py",
                            "content_step": 1
                        }
                    }
                ],
                "planning_rationale": "Use the locked explicit path."
            }

        def invoke_text(self, messages, stream=False):
            return "print('hello world')"

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceConfig(root=str(workspace_root))
    llm = BadThenValidPathLLM()

    tool_registry = build_tool_registry(
        llm_client=llm,
        workspace=workspace
    )

    coordinator = Coordinator(
        planner=PlannerAgent(llm_client=llm, workspace_config=workspace),
        executor=ExecutorAgent(tool_registry=tool_registry),
        reviewer=FakeReviewer(),
        memory=FakeMemory(),
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm,
        router=FakeRouter()
    )

    message = coordinator.start_workflow(
        conversation_id=1,
        user_task="Create src/main.py with a simple Python hello world program"
    )

    assert message.message_type == "permission_request"
    assert len(llm.json_calls) == 2

    requested_tools = message.response["requested_tools"]
    assert len(requested_tools) == 1
    assert requested_tools[0]["tool"] == "create_file"
    assert requested_tools[0]["resolved_args"]["path"] == "src/main.py"
    assert requested_tools[0]["resolved_args"]["content"] == "print('hello world')"

    plan_steps = message.response["plan_response"]["steps"]
    mutation_paths = [
        step["args"].get("path")
        for step in plan_steps
        if step["tool"] in {"create_dir", "create_file", "write_file", "append_file", "replace_text"}
    ]
    assert "code.py" not in mutation_paths
    assert mutation_paths == ["src/main.py"]


def test_execution_verifier_rejects_before_reviewer_acceptance(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    workspace = WorkspaceConfig(root=str(workspace_root))
    llm = FakeLLM()
    tool_registry = build_tool_registry(
        llm_client=llm,
        workspace=workspace
    )
    reviewer = FakeReviewer()
    memory = FakeMemory()

    coordinator = Coordinator(
        planner=FakePlanner(),
        executor=ExecutorAgent(tool_registry=tool_registry),
        reviewer=reviewer,
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm,
        router=FakeRouter(),
        execution_verifier=ExecutionVerifier(workspace_config=workspace)
    )

    executor_response = {
        "goal": "Create work.txt with hello",
        "approved_actions": [],
        "rollback_log": [],
        "workspace_before": [],
        "step_results": {
            "1": "File 'work.txt' created successfully."
        },
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {
                    "path": "work.txt",
                    "content": "hello"
                },
                "resolved_args": {
                    "path": "work.txt",
                    "content": "hello"
                },
                "result": "File 'work.txt' created successfully."
            }
        ]
    }
    executor_msg = Message(
        conversation_id=1,
        step_index=4,
        sender="executor",
        receiver="coordinator",
        target_agent=None,
        message_type="execution_result",
        status="completed",
        response=executor_response,
        visibility="internal"
    )

    message = coordinator.continue_workflow(
        conversation_id=1,
        user_task="create work.txt with hello",
        plan_response={
            "goal": "Create work.txt with hello",
            "steps": [
                {
                    "id": 1,
                    "tool": "create_file",
                    "args": {
                        "path": "work.txt",
                        "content": "hello"
                    }
                }
            ],
            "route": {"task_type": "workspace_mutation"}
        },
        executor_msg=executor_msg,
        review_retry_used=True
    )

    assert message.status == "failed"
    assert message.message_type == "workflow_result"
    assert any("work.txt" in issue for issue in message.response["issues"])
    assert reviewer.calls == []
