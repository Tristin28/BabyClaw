"""
Testing out the BabyClaw pipeline.
This runner:
- prints permission requests clearly
- prints compiled plans
- prints execution traces
- avoids printing raw Message(...) objects
"""

from src.Agents.Planning.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.ReviewerAgent import ReviewerAgent
from src.Agents.Coordinator import Coordinator
from src.OllamaClient import OllamaClient
from src.Memory.sql_database import DatabaseManager
from src.tools.tool_registry import build_tool_registry
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.message import Message
from src.tools.utils import WorkspaceConfig
from src.config.workspace_config import load_workspace_path, save_workspace_path


def print_indent(value, indent: str = "    "):
    text = str(value)

    for line in text.splitlines():
        print(f"{indent}{line}")


def render_plan(plan_response: dict):
    if not plan_response:
        return

    steps = plan_response.get("steps", [])

    if not steps:
        return

    print("\nCompiled plan:")

    for step in steps:
        print(
            f"- Step {step.get('id')} | "
            f"Tool: {step.get('tool')} | "
            f"Args: {step.get('args')} | "
            f"Depends on: {step.get('depends_on', [])}"
        )


def render_execution_trace(trace: list[dict]):
    if not trace:
        print("\nExecution trace: None")
        return

    print("\nExecution trace:")

    for step in trace:
        step_id = step.get("id", "?")
        tool = step.get("tool", "unknown_tool")
        status = step.get("status", "unknown")

        print(f"\nStep {step_id}")
        print(f"  Tool: {tool}")
        print(f"  Status: {status}")

        if "args" in step:
            print(f"  Args: {step['args']}")

        if "depends_on" in step:
            print(f"  Depends on: {step['depends_on']}")

        if "result" in step:
            print("  Result:")
            print_indent(step["result"])

        if "error" in step:
            print("  Error:")
            print_indent(step["error"])


def render_message(msg: Message):
    print()
    print("=" * 70)

    response = msg.response or {}

    if msg.message_type == "permission_request":
        print("System: Permission is required before continuing.\n")

        print("Requested tools:")
        for tool in response.get("requested_tools", []):
            print(f"- Step {tool.get('step_id')} | Tool: {tool.get('tool')}")
            print(f"  Args: {tool.get('args')}")
            print(f"  Description: {tool.get('description')}")

        render_plan(response.get("plan_response", {}))

        print("\nReply with 'yes' to approve or 'no' to cancel.")
        print("=" * 70)
        return

    if msg.status == "failed":
        print("System: Task failed.\n")

        if "message" in response:
            print(f"Message: {response['message']}")

        if "error" in response:
            print(f"Reason: {response['error']}")

        if "review_summary" in response:
            print(f"Review: {response['review_summary']}")

        issues = response.get("issues", [])
        if issues:
            print("\nIssues:")
            for issue in issues:
                print(f"- {issue}")

        execution_state = response.get("execution_state")
        if execution_state:
            render_execution_trace(execution_state.get("execution_trace", []))

        print("=" * 70)
        return

    if msg.status == "cancelled":
        print(f"System: {response.get('message', 'Task cancelled.')}")
        print("=" * 70)
        return

    if msg.status == "completed":
        print(f"System: {response.get('message', 'Task completed successfully.')}")

        if "review_summary" in response:
            print(f"\nReview: {response['review_summary']}")

        render_execution_trace(response.get("execution_trace", []))

        print("=" * 70)
        return

    print("System: Received response.")
    print(response)
    print("=" * 70)


def print_suggested_tests():
    print("Suggested tests:")
    print("1. append to hello.txt by saying hey")
    print("2. read hello.txt")
    print("3. summarise hello.txt")
    print("4. create a file called Start.txt and inside it write hello")
    print("5. summarise the start file")
    print("6. read hello file and summarise it")
    print()


def main():
    llm_client = OllamaClient(model="qwen2.5:3b")

    workspace = WorkspaceConfig(load_workspace_path())
    tool_registry = build_tool_registry(llm_client, workspace)

    db_manager = DatabaseManager("memory.db")
    db_manager.init_db()

    planner = PlannerAgent(llm_client=llm_client)
    executor = ExecutorAgent(tool_registry=tool_registry)
    reviewer = ReviewerAgent(llm_client=llm_client)
    memory = MemoryAgent(db_manager=db_manager, llm_client=llm_client)

    coordinator = Coordinator(
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm_client
    )

    conversation_id = 1
    pending_permission = None

    print("BabyClaw is ready.")
    print("Type a task, or type 'exit' to stop.")
    print("Type 'set workspace <path>' to change workspace.\n")

    print_suggested_tests()

    while True:
        user_input = input("You: ").strip()

        if user_input == "":
            continue

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        if user_input.startswith("set workspace "):
            new_path = user_input.replace("set workspace ", "").strip()

            try:
                workspace.set_root(new_path)
                save_workspace_path(new_path)
                print(f"Workspace changed to: {workspace.root}")
            except Exception as e:
                print(f"Could not change workspace: {e}")

            continue

        if pending_permission is not None:
            if user_input.lower() not in ["yes", "no"]:
                print("System: Please answer the pending permission request with 'yes' or 'no'.")
                continue

            approved = user_input.lower() == "yes"

            msg = coordinator.continue_after_permission(
                conversation_id=conversation_id,
                user_task=pending_permission["user_task"],
                plan_response=pending_permission["plan_response"],
                execution_state=pending_permission["execution_state"],
                pending_runnable_steps=pending_permission["pending_runnable_steps"],
                step_index=pending_permission["next_step_index"],
                approved=approved
            )

            pending_permission = None

        else:
            msg = coordinator.start_workflow(
                conversation_id=conversation_id,
                user_task=user_input
            )

        if msg.status == "waiting" and msg.message_type == "permission_request":
            pending_permission = msg.response
            render_message(msg)
            continue

        render_message(msg)

        if msg.status in ["completed", "failed", "cancelled"]:
            pending_permission = None


if __name__ == "__main__":
    main()