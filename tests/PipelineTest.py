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
from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent
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

def print_section(title: str):
    print()
    print(title)


def render_key_value(title: str, value):
    if value is None:
        return

    print(f"\n{title}:")
    print_indent(value)


def render_debug_list(title: str, values):
    print_section(title)

    if not values:
        print("None")
        return

    for value in values:
        print(f"- {value}")


def render_display_result(response: dict):
    display_result = response.get("display_result")
    direct_response = response.get("direct_response")

    if display_result:
        print_section("DISPLAY RESULT SHOWN TO USER")
        print_indent(display_result)
        return

    if direct_response:
        print_section("DIRECT RESPONSE")
        print_indent(direct_response)
        return

    print_section("DISPLAY RESULT SHOWN TO USER")
    print("None")


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
    print_section("EXECUTION TRACE")

    if not trace:
        print("None")
        return

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
            print_indent(step["result"], indent="    ")

        if "error" in step:
            print("  Error:")
            print_indent(step["error"], indent="    ")

def render_rollback_log(rollback_log: list[dict]):
    print_section("ROLLBACK LOG")

    if not rollback_log:
        print("None")
        return

    for entry in rollback_log:
        print(f"\nStep {entry.get('step_id')}")
        print(f"  Tool: {entry.get('tool')}")
        print(f"  Resolved args: {entry.get('resolved_args')}")
        print(f"  Snapshot:")
        print_indent(entry.get("snapshot"))


def render_rollback_results(rollback_results):
    print_section("ROLLBACK RESULTS")

    if not rollback_results:
        print("None")
        return

    for result in rollback_results:
        print(f"- Step {result.get('step_id')} | Tool: {result.get('tool')} | Status: {result.get('status')}")

        if "error" in result:
            print("  Error:")
            print_indent(result["error"])


def render_approved_actions(approved_actions):
    print_section("APPROVED ACTIONS")

    if not approved_actions:
        print("None")
        return

    for action in approved_actions:
        print(f"- {action}")


def render_step_results(step_results):
    print_section("STEP RESULTS")

    if not step_results:
        print("None")
        return

    for step_id, result in step_results.items():
        print(f"\nStep {step_id}:")
        print_indent(result)


def render_message(msg: Message):
    print()

    response = msg.response or {}

    print("MESSAGE DEBUG")
    print(f"Status: {msg.status}")
    print(f"Message type: {msg.message_type}")
    print(f"Sender: {msg.sender}")
    print(f"Receiver: {msg.receiver}")
    print(f"Target agent: {msg.target_agent}")
    print(f"Response keys: {list(response.keys())}")

    if msg.message_type == "permission_request":
        print_section("PERMISSION REQUEST")
        print("System: Permission is required before continuing.")

        print_section("REQUESTED TOOLS")
        for tool in response.get("requested_tools", []):
            print(f"- Step {tool.get('step_id')} | Tool: {tool.get('tool')}")
            print(f"  Args: {tool.get('args')}")
            print(f"  Description: {tool.get('description')}")

        render_plan(response.get("plan_response", {}))

        execution_state = response.get("execution_state", {})
        render_execution_trace(execution_state.get("execution_trace", []))
        render_approved_actions(execution_state.get("approved_actions", []))
        render_rollback_log(execution_state.get("rollback_log", []))

        print_section("NEXT ACTION")
        print("Reply with 'yes' to approve or 'no' to cancel.")
        return

    if msg.status == "failed":
        print_section("FINAL STATUS")
        print("System: Task failed.")

        render_key_value("Message", response.get("message"))
        render_key_value("Reason", response.get("error"))
        render_key_value("Review", response.get("review_summary"))

        issues = response.get("issues", [])
        if issues:
            print_section("ISSUES")
            for issue in issues:
                print(f"- {issue}")

        render_display_result(response)

        execution_state = response.get("execution_state")
        if execution_state:
            render_execution_trace(execution_state.get("execution_trace", []))
            render_step_results(execution_state.get("step_results", {}))
            render_approved_actions(execution_state.get("approved_actions", []))
            render_rollback_log(execution_state.get("rollback_log", []))

        render_rollback_results(response.get("rollback_results", []))

        return

    if msg.status == "cancelled":
        print_section("FINAL STATUS")
        print(f"System: {response.get('message', 'Task cancelled.')}")
        print("=" * 70)
        return

    if msg.status == "completed":
        print_section("FINAL STATUS")
        print(f"System: {response.get('message', 'Task completed successfully.')}")

        render_display_result(response)

        if "review_summary" in response:
            render_key_value("Review", response["review_summary"])

        render_plan(response.get("plan_response", {}))
        render_execution_trace(response.get("execution_trace", []))
        render_step_results(response.get("step_results", {}))
        render_approved_actions(response.get("approved_actions", []))
        render_rollback_log(response.get("rollback_log", []))
        render_rollback_results(response.get("rollback_results", []))

        print("=" * 70)
        return

    print_section("RAW RESPONSE")
    print("System: Received response.")
    print(response)
    print("=" * 70)

    
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