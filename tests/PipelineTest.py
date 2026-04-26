from src.Agents.Planning.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent
from src.Agents.Coordinator import Coordinator

from src.OllamaClient import OllamaClient
from src.Memory.sql_database import DatabaseManager
from src.tools.tool_registry import build_tool_registry
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.utils import WorkspaceConfig


CONVERSATION_ID = 1


def build_system():
    llm_client = OllamaClient()
    db_manager = DatabaseManager("./memory.db")
    workspace = WorkspaceConfig(root="workspace")

    tool_registry = build_tool_registry(
        llm_client=llm_client,
        workspace=workspace
    )

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

    return coordinator, workspace


def display_result(message):
    response = message.response or {}

    if message.message_type == "permission_request":
        print("\nSystem: This task needs permission before modifying the workspace.")

        requested_tools = response.get("requested_tools", [])

        for tool in requested_tools:
            print(f"- {tool.get('tool')} with args {tool.get('args')}")

        print("\nReply with 'yes' to approve or 'no' to cancel.")
        return

    if message.status == "cancelled":
        print("\nSystem:", response.get("message", "Task cancelled."))
        return

    if message.status == "completed":
        output = (
            response.get("direct_response")
            or response.get("display_result")
            or response.get("message")
            or "Task completed successfully."
        )

        print("\nSystem:")
        print(output)
        return

    output = response.get("message", "Task could not be completed successfully.")
    print("\nSystem:")
    print(output)

    issues = response.get("issues", [])
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"- {issue}")


def main():
    coordinator, workspace = build_system()
    pending_permission = None

    print("BabyClaw is ready.")
    print("Type a task, or type 'exit' to stop.")
    print("Type 'set workspace <path>' to change workspace.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        if not user_input:
            continue

        if user_input.lower().startswith("set workspace "):
            new_workspace = user_input[len("set workspace "):].strip()

            try:
                workspace.set_root(new_workspace)
                print(f"System: Workspace changed to {workspace.root}")
            except Exception as e:
                print(f"System: Failed to change workspace: {e}")

            continue

        if pending_permission:
            if user_input.lower() not in {"yes", "y", "no", "n"}:
                print("System: Please reply with 'yes' or 'no'.")
                continue

            approved = user_input.lower() in {"yes", "y"}

            message = coordinator.continue_after_permission(
                conversation_id=CONVERSATION_ID,
                user_task=pending_permission["user_task"],
                plan_response=pending_permission["plan_response"],
                execution_state=pending_permission["execution_state"],
                pending_runnable_steps=pending_permission["pending_runnable_steps"],
                step_index=pending_permission["next_step_index"],
                approved=approved
            )

            pending_permission = None

            display_result(message)

            if message.message_type == "permission_request":
                pending_permission = message.response

            continue

        message = coordinator.start_workflow(
            conversation_id=CONVERSATION_ID,
            user_task=user_input
        )

        display_result(message)

        if message.message_type == "permission_request":
            pending_permission = message.response


if __name__ == "__main__":
   main()