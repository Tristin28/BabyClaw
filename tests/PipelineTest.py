'''
    Testing out the respective pipeline with generated code
'''
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


def render_message(msg: Message):
    print()

    if msg.message_type == "permission_request":
        print("System: Permission required.")
        for tool in msg.response["requested_tools"]:
            print(f"- Step {tool['step_id']} | {tool['tool']}")

    elif msg.message_type == "execution_wave_result":
        print("System: Execution wave completed.")

    elif msg.message_type == "execution_failed":
        print("System: Execution failed.")
        print(msg.response)

    elif msg.message_type == "workflow_result":

        print(f"System: {msg.response.get('message','')}")

        if "review_summary" in msg.response:
            print(f"Review: {msg.response['review_summary']}")

        if "issues" in msg.response:
            for issue in msg.response["issues"]:
                print(f"- {issue}")

        if "execution_trace" in msg.response:

            print("\nExecution trace:")

            for step in msg.response["execution_trace"]:

                print(
                    f"- Step {step['id']} | "
                    f"{step['tool']} | "
                    f"{step['status']}"
                )

                if "result" in step:
                    print("  Output:")
                    print(step["result"])


def main():
    '''
        Main application entry point.
        This is what lets the user communicate with the agent system.
    '''
    llm_client = OllamaClient(model="qwen2.5:3b")
    workspace = WorkspaceConfig(load_workspace_path())
    tool_registry = build_tool_registry(llm_client, workspace)
    db_manager = DatabaseManager("memory.db")
    db_manager.init_db()

    planner_tool_descriptions = PLANNER_TOOL_DESCRIPTIONS

    planner = PlannerAgent(llm_client=llm_client)
    executor = ExecutorAgent(tool_registry=tool_registry)
    reviewer = ReviewerAgent(llm_client=llm_client)
    memory = MemoryAgent(db_manager=db_manager, llm_client=llm_client)

    coordinator = Coordinator(planner=planner, executor=executor, reviewer=reviewer, memory=memory,
                              planner_tool_descriptions=planner_tool_descriptions, tool_registry=tool_registry,
                              llm_client=llm_client)

    conversation_id = 1

    #This stores a paused workflow when permission is needed
    pending_permission = None

    print("BabyClaw is ready.")
    print("Type a task, or type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.startswith("set workspace "):
            new_path = user_input.replace("set workspace ", "").strip()

            try:
                workspace.set_root(new_path)
                save_workspace_path(new_path)
                print(f"Workspace changed to: {workspace.root}")
            except Exception as e:
                print(f"Could not change workspace: {e}")

            continue

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        if user_input == "":
            continue

        #If there is a pending permission request, treat the user's input as yes/no
        if pending_permission is not None:
            if user_input.lower() not in ["yes", "no"]:
                print("System: Previous pending task cancelled. Starting new task instead.")
                pending_permission = None
                print("Please answer with 'yes' or 'no'.")
                continue

            approved = user_input.lower() == "yes"

            msg = coordinator.continue_after_permission(conversation_id=conversation_id,
                                                        user_task=pending_permission["user_task"],
                                                        plan_response=pending_permission["plan_response"],
                                                        execution_state=pending_permission["execution_state"],
                                                        pending_runnable_steps=pending_permission["pending_runnable_steps"],
                                                        step_index=pending_permission["next_step_index"],
                                                        approved=approved)

            pending_permission = None

        else:
            msg = coordinator.start_workflow(conversation_id=conversation_id, user_task=user_input)

        #If coordinator needs permission, save the workflow state and ask user
        if msg.status == "waiting" and msg.message_type == "permission_request":
            pending_permission = msg.response

            print("\nSystem: Permission is required before continuing.")
            print("Requested tools:")

            for tool in msg.response["requested_tools"]:
                print(f"\n- Step {tool['step_id']} | Tool: {tool['tool']} | {tool['description']}")

            print("\nReply with 'yes' to approve or 'no' to cancel.\n")
            continue

        #Otherwise print final workflow result
        render_message(msg)
        print()

        #If workflow fully ended, clear any saved permission state
        if msg.status in ["completed", "failed", "cancelled"]:
            pending_permission = None


if __name__ == "__main__":
    main()