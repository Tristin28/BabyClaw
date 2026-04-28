from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent
from src.Agents.Coordinator import Coordinator
from src.message import Message
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

    def store_long_term_memory(self, user_task: str, episode_summary: str, conversation_id: int, step_index: int):
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
            execution_response: dict, workspace_before=None, workspace_after=None):

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

    coordinator = Coordinator(
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm
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
    from src.Agents.Planning.PlannerAgent import PlannerAgent

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