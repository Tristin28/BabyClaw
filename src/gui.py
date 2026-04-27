"""
    Flask-based GUI for BabyClaw.

    This is a thin web layer over the same Coordinator that src/Main.py drives
    from the CLI. It exposes a chat interface in the browser so the user can
    talk to the agent system, approve or deny permission requests, and change
    the workspace folder.

    Run with:
        python -m src.gui

    Then open http://127.0.0.1:5000
"""

from pathlib import Path
import threading

from flask import Flask, jsonify, render_template, request

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
MEMORY_DIR = PROJECT_ROOT / "Memory"
DB_PATH = MEMORY_DIR / "memory.db"


def build_system():
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    llm_client = OllamaClient(model="qwen2.5-coder:7b")

    db_manager = DatabaseManager(db_path=str(DB_PATH))
    db_manager.init_db()

    workspace = WorkspaceConfig(root=str(WORKSPACE_DIR))

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

def _jsonable(value):
    """Recursively convert sets/tuples to lists so jsonify can handle the value."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_jsonable(v) for v in value]
    return value

def message_to_payload(message) -> dict:
    """
        Convert a Message object into a JSON-serialisable shape that the
        browser knows how to render.
    """
    response = message.response or {}

    payload = {
        "message_type": message.message_type,
        "status": message.status,
        "step_index": message.step_index,
        "sender": message.sender,
        "receiver": message.receiver,
        "timestamp": message.timestamp,
    }

    if message.message_type == "permission_request":
        tools = []
        for tool in response.get("requested_tools", []):
            tools.append({
                "tool": tool.get("tool"),
                "args": tool.get("args"),
            })

        payload["kind"] = "permission_request"
        payload["text"] = "This task needs permission before modifying the workspace."
        payload["requested_tools"] = tools
        return _jsonable(payload)

    if message.status == "cancelled":
        payload["kind"] = "cancelled"
        payload["text"] = response.get("message", "Task cancelled.")
        return _jsonable(payload)

    if message.status == "completed":
        payload["kind"] = "completed"
        payload["text"] = (
            response.get("direct_response")
            or response.get("display_result")
            or response.get("message")
            or "Task completed successfully."
        )
        return _jsonable(payload)

    payload["kind"] = "failed"
    payload["text"] = response.get("message", "Task could not be completed successfully.")
    payload["issues"] = response.get("issues", [])
    return _jsonable(payload)

COORDINATOR, WORKSPACE = build_system()
PENDING_PERMISSION = None
WORKFLOW_LOCK = threading.Lock()

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def state():
    return jsonify({
        "workspace": str(WORKSPACE.root),
        "awaiting_permission": PENDING_PERMISSION is not None,
    })


@app.route("/api/workspace", methods=["POST"])
def set_workspace():
    data = request.get_json(silent=True) or {}
    new_root = (data.get("path") or "").strip()

    if not new_root:
        return jsonify({"ok": False, "error": "Path is required."}), 400

    try:
        WORKSPACE.set_root(new_root)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "workspace": str(WORKSPACE.root)})


@app.route("/api/message", methods=["POST"])
def send_message():
    """
        Run a new task through the coordinator, or feed a yes/no answer into
        a pending permission request that is already in progress.
    """
    global PENDING_PERMISSION

    data = request.get_json(silent=True) or {}
    user_input = (data.get("text") or "").strip()

    if not user_input:
        return jsonify({"ok": False, "error": "Empty message."}), 400

    with WORKFLOW_LOCK:
        if PENDING_PERMISSION is not None:
            answer = user_input.lower()

            if answer not in {"yes", "y", "no", "n"}:
                return jsonify({
                    "ok": False,
                    "error": "A permission request is pending. Reply with 'yes' or 'no'.",
                }), 400

            approved = answer in {"yes", "y"}
            pending = PENDING_PERMISSION

            message = COORDINATOR.continue_after_permission(
                conversation_id=CONVERSATION_ID,
                user_task=pending["user_task"],
                plan_response=pending["plan_response"],
                execution_state=pending["execution_state"],
                pending_runnable_steps=pending["pending_runnable_steps"],
                step_index=pending["next_step_index"],
                approved=approved,
            )

            PENDING_PERMISSION = None
        else:
            message = COORDINATOR.start_workflow(
                conversation_id=CONVERSATION_ID,
                user_task=user_input,
            )

        if message.message_type == "permission_request":
            PENDING_PERMISSION = message.response

    return jsonify({
        "ok": True,
        "awaiting_permission": PENDING_PERMISSION is not None,
        "message": message_to_payload(message),
    })


@app.route("/api/permission", methods=["POST"])
def respond_to_permission():
    """
        Dedicated endpoint for the Yes / No buttons in the GUI, so the user
        does not have to type 'yes' or 'no' into the chat box.
    """
    global PENDING_PERMISSION

    data = request.get_json(silent=True) or {}
    approved = bool(data.get("approved"))

    with WORKFLOW_LOCK:
        if PENDING_PERMISSION is None:
            return jsonify({"ok": False, "error": "No permission request is pending."}), 400

        pending = PENDING_PERMISSION

        message = COORDINATOR.continue_after_permission(
            conversation_id=CONVERSATION_ID,
            user_task=pending["user_task"],
            plan_response=pending["plan_response"],
            execution_state=pending["execution_state"],
            pending_runnable_steps=pending["pending_runnable_steps"],
            step_index=pending["next_step_index"],
            approved=approved,
        )

        PENDING_PERMISSION = None

        if message.message_type == "permission_request":
            PENDING_PERMISSION = message.response

    return jsonify({
        "ok": True,
        "awaiting_permission": PENDING_PERMISSION is not None,
        "message": message_to_payload(message),
    })


def main():
    print("\nBabyClaw GUI is starting.")
    print(f"Workspace: {WORKSPACE.root}")
    print(f"Memory DB: {DB_PATH}")
    print("Open http://127.0.0.1:5000 in your browser.\n")

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()