# src/cli.py

from pathlib import Path
import json
import sqlite3
from pprint import pprint

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

# cli.py is inside:
# BabyClaw/src/cli.py
#
# parents[0] = src
# parents[1] = BabyClaw
PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# This stores the SQLite DB inside the BabyClaw folder.
# Example:
# BabyClaw/Memory/memory.db
#
# Note:
# This is a runtime storage folder.
# It is different from src/Memory, which contains Python code.
MEMORY_DIR = PROJECT_ROOT / "Memory"

DB_PATH = MEMORY_DIR / "memory.db"


# Set this to True when debugging.
# Set to False when you want normal clean output.
DEBUG_MODE = True


def debug_print(title: str, value=None):
    """
        Helper used to print debugging information in a consistent way.

        This lets us see what paths are being used, what DB is being loaded,
        and what messages are being returned by the Coordinator.
    """
    if not DEBUG_MODE:
        return

    print(f"\n[DEBUG] {title}")

    if value is not None:
        pprint(value)


def ensure_project_dirs():
    """
        Creates the folders BabyClaw needs inside the project folder.

        This means the runtime files will stay visible inside:

            BabyClaw/workspace/
            BabyClaw/Memory/
            BabyClaw/Memory/memory.db
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def debug_paths():
    """
        Shows where BabyClaw is reading/writing runtime files.
    """
    print("\n========== DEBUG PATHS ==========")
    print(f"PROJECT_ROOT:  {PROJECT_ROOT}")
    print(f"WORKSPACE_DIR: {WORKSPACE_DIR}")
    print(f"MEMORY_DIR:    {MEMORY_DIR}")
    print(f"DB_PATH:       {DB_PATH}")
    print(f"DB exists:     {DB_PATH.exists()}")
    print("=================================")


def load_recent_db_messages(limit: int = 10):
    """
        Reads recent messages directly from memory.db.

        This is only for debugging.
        It lets you confirm what is actually stored in SQLite.
    """
    if not DB_PATH.exists():
        print("\n[DEBUG] memory.db does not exist yet.")
        return []

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                id,
                conversation_id,
                step_index,
                sender,
                receiver,
                target_agent,
                message_type,
                status,
                response,
                visibility,
                timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (CONVERSATION_ID, limit)
        ).fetchall()

    return list(rows)


def debug_recent_messages(limit: int = 10):
    """
        Prints the most recent messages in a readable way.

        This helps you check whether the user's name, previous task,
        planner result, executor result, and reviewer result are actually
        being stored.
    """
    rows = load_recent_db_messages(limit=limit)

    print(f"\n========== LAST {limit} DB MESSAGES ==========")

    if not rows:
        print("No messages found.")
        print("=============================================")
        return

    for row in reversed(rows):
        print("\n---------------------------------------------")
        print(f"id:           {row['id']}")
        print(f"timestamp:    {row['timestamp']}")
        print(f"step_index:   {row['step_index']}")
        print(f"sender:       {row['sender']}")
        print(f"receiver:     {row['receiver']}")
        print(f"target_agent: {row['target_agent']}")
        print(f"type:         {row['message_type']}")
        print(f"status:       {row['status']}")
        print(f"visibility:   {row['visibility']}")

        try:
            response = json.loads(row["response"])
        except Exception:
            response = row["response"]

        print("response:")
        pprint(response)

    print("\n=============================================")


def debug_context_like_view(limit: int = 10):
    """
        Prints a simplified context view from recent SQL messages.

        This is useful because your agents do not use the raw DB file directly.
        The Coordinator / repositories usually fetch recent messages and pass
        them into prompts as context.

        This view helps you see what useful information exists in recent history.
    """
    rows = load_recent_db_messages(limit=limit)

    print(f"\n========== CONTEXT-LIKE VIEW FROM LAST {limit} MESSAGES ==========")

    if not rows:
        print("No messages found.")
        print("===============================================================")
        return

    for row in reversed(rows):
        try:
            response = json.loads(row["response"])
        except Exception:
            response = {}

        content = ""

        if isinstance(response, dict):
            content = (
                response.get("content")
                or response.get("direct_response")
                or response.get("display_result")
                or response.get("message")
                or response.get("goal")
                or ""
            )

            if not content and "step_results" in response:
                content = str(response.get("step_results"))

        print(f"\n{row['sender']} -> {row['receiver']} | {row['message_type']} | {row['status']}")
        print(f"content/context text: {content}")

    print("\n===============================================================")


def debug_message_object(message):
    """
        Prints the final Message object returned by the Coordinator.

        This helps you see whether the bug came from:
            - Planner
            - Executor
            - Reviewer
            - Coordinator final formatting
    """
    if not DEBUG_MODE:
        return

    print("\n========== RETURNED MESSAGE DEBUG ==========")
    print(f"conversation_id: {message.conversation_id}")
    print(f"step_index:      {message.step_index}")
    print(f"sender:          {message.sender}")
    print(f"receiver:        {message.receiver}")
    print(f"target_agent:    {message.target_agent}")
    print(f"message_type:    {message.message_type}")
    print(f"status:          {message.status}")
    print(f"visibility:      {message.visibility}")
    print("response:")
    pprint(message.response)
    print("============================================")


def build_system():
    """
        Builds the full BabyClaw system.

        This version stores the SQLite database inside:

            BabyClaw/Memory/memory.db

        It also prints the paths being used when DEBUG_MODE is True.
    """
    ensure_project_dirs()

    debug_paths()

    llm_client = OllamaClient()

    db_manager = DatabaseManager(
        db_path=str(DB_PATH)
    )

    # This creates memory.db and the messages table if they do not exist yet.
    db_manager.init_db()

    debug_print("Database initialised", {
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists()
    })

    workspace = WorkspaceConfig(
        root=str(WORKSPACE_DIR)
    )

    debug_print("Workspace configured", {
        "workspace_root": str(WORKSPACE_DIR)
    })

    tool_registry = build_tool_registry(
        llm_client=llm_client,
        workspace=workspace
    )

    debug_print("Tool registry loaded", {
        "tools": list(tool_registry.keys())
    })

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

    debug_print("BabyClaw system built successfully")

    return coordinator, workspace


def display_result(message):
    response = message.response or {}

    debug_message_object(message)

    if message.message_type == "permission_request":
        print("\nSystem: This task needs permission before modifying the workspace.")

        for tool in response.get("requested_tools", []):
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

    print("\nSystem:")
    print(response.get("message", "Task could not be completed successfully."))

    issues = response.get("issues", [])
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"- {issue}")


def print_help():
    print(
        """
BabyClaw commands:

  exit
      Stop BabyClaw.

  set workspace <path>
      Change the workspace folder.

  debug paths
      Show project root, workspace path, memory folder, and memory.db path.

  debug recent
      Show the last 10 raw messages stored inside SQLite.

  debug recent <number>
      Show the last <number> raw messages stored inside SQLite.

  debug context
      Show a simplified context-like view from the recent SQL messages.

  debug context <number>
      Show a simplified context-like view from the last <number> SQL messages.

  debug on
      Turn debug printing on.

  debug off
      Turn debug printing off.

  help
      Show this help menu.
        """
    )


def handle_debug_command(user_input: str) -> bool:
    """
        Handles debug commands.

        Returns True if the command was handled.
        Returns False if this was not a debug command.
    """
    global DEBUG_MODE

    command = user_input.strip().lower()

    if command == "help":
        print_help()
        return True

    if command == "debug on":
        DEBUG_MODE = True
        print("System: Debug mode is now ON.")
        return True

    if command == "debug off":
        DEBUG_MODE = False
        print("System: Debug mode is now OFF.")
        return True

    if command == "debug paths":
        debug_paths()
        return True

    if command.startswith("debug recent"):
        parts = command.split()

        limit = 10

        if len(parts) == 3:
            try:
                limit = int(parts[2])
            except ValueError:
                print("System: Please use a number, for example: debug recent 20")
                return True

        debug_recent_messages(limit=limit)
        return True

    if command.startswith("debug context"):
        parts = command.split()

        limit = 10

        if len(parts) == 3:
            try:
                limit = int(parts[2])
            except ValueError:
                print("System: Please use a number, for example: debug context 20")
                return True

        debug_context_like_view(limit=limit)
        return True

    return False


def main():
    coordinator, workspace = build_system()
    pending_permission = None

    print("\nBabyClaw is ready.")
    print("Type a task, or type 'exit' to stop.")
    print("Type 'set workspace <path>' to change workspace.")
    print("Type 'help' to see debugging commands.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        if not user_input:
            continue

        if handle_debug_command(user_input):
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

            debug_print("Continuing after permission", {
                "approved": approved,
                "pending_permission": pending_permission
            })

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

        debug_print("Starting workflow", {
            "conversation_id": CONVERSATION_ID,
            "user_task": user_input
        })

        message = coordinator.start_workflow(
            conversation_id=CONVERSATION_ID,
            user_task=user_input
        )

        display_result(message)

        if message.message_type == "permission_request":
            pending_permission = message.response


if __name__ == "__main__":
    main()