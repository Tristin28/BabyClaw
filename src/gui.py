"""
    Flask-based GUI for BabyClaw.

    Thin web layer over the same Coordinator that src/Main.py drives from
    the CLI. Adds tabs that show:
        - Chat
        - Plans (recent planner outputs)
        - Memory (vector memories + recent SQL messages)
        - Settings (workspace + memory paths)

    The workspace path is persisted to config/workspace_config.json so that
    a path the user picks survives between runs.

    Run with:
        python -m src.gui

    The browser opens automatically at http://127.0.0.1:5000.
"""

from pathlib import Path
import json
import sqlite3
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request

from src.config.workspace_config import save_workspace_path, load_workspace_path

from src.Agents.Planning.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent
from src.Agents.Coordinator import Coordinator
from src.Agents.Routing.RouteAgent import RouteAgent

from src.OllamaClient import OllamaClient
from src.Memory.sql_database import DatabaseManager
from src.tools.tool_registry import build_tool_registry
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.utils import WorkspaceConfig


CONVERSATION_ID = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = PROJECT_ROOT / "Memory"
DB_PATH = MEMORY_DIR / "memory.db"
VECTOR_DIR = MEMORY_DIR / "chroma_db"


def build_system():
    workspace_path = Path(load_workspace_path())
    workspace_path.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    llm_client = OllamaClient()

    db_manager = DatabaseManager(db_path=str(DB_PATH))
    db_manager.init_db()

    workspace = WorkspaceConfig(root=str(workspace_path))

    tool_registry = build_tool_registry(
        llm_client=llm_client,
        workspace=workspace
    )

    planner = PlannerAgent(llm_client=llm_client, workspace_config=workspace)
    executor = ExecutorAgent(tool_registry=tool_registry)
    reviewer = ReviewerAgent(llm_client=llm_client, workspace_config=workspace)
    memory = MemoryAgent(db_manager=db_manager, llm_client=llm_client)
    router = RouteAgent(llm_client=llm_client)

    coordinator = Coordinator(
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        memory=memory,
        planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS,
        tool_registry=tool_registry,
        llm_client=llm_client,
        router=router
    )

    return coordinator, workspace, memory


def _jsonable(value):
    """
        Recursively convert sets/tuples to lists so jsonify can handle the value.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_jsonable(v) for v in value]
    return value


def message_to_payload(message) -> dict:
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


def load_recent_db_rows(limit: int = 50) -> list[dict]:
    if not DB_PATH.exists():
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

    items = []
    for row in rows:
        try:
            response = json.loads(row["response"])
        except Exception:
            response = {}

        items.append({
            "id": row["id"],
            "step_index": row["step_index"],
            "sender": row["sender"],
            "receiver": row["receiver"],
            "target_agent": row["target_agent"],
            "message_type": row["message_type"],
            "status": row["status"],
            "visibility": row["visibility"],
            "timestamp": row["timestamp"],
            "response": response,
        })

    return items


COORDINATOR, WORKSPACE, MEMORY_AGENT = build_system()
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
        "memory_db": str(DB_PATH),
        "vector_db": str(VECTOR_DIR),
        "project_root": str(PROJECT_ROOT),
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
        save_workspace_path(WORKSPACE.root)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "workspace": str(WORKSPACE.root)})


@app.route("/api/plans", methods=["GET"])
def get_plans():
    rows = load_recent_db_rows(limit=200)

    plans = []
    for row in rows:
        if row["sender"] != "planner":
            continue

        response = row["response"] if isinstance(row["response"], dict) else {}

        plans.append({
            "timestamp": row["timestamp"],
            "step_index": row["step_index"],
            "status": row["status"],
            "goal": response.get("goal", ""),
            "steps": response.get("steps", []),
            "rationale": response.get("planning_rationale", ""),
            "error": response.get("error", ""),
        })

    return jsonify({"plans": plans})


@app.route("/api/memory", methods=["GET"])
def get_memory():
    vector_items = []
    try:
        vector_data = MEMORY_AGENT.vector_repo.get_all_memories() or {}

        documents = vector_data.get("documents") or []
        metadatas = vector_data.get("metadatas") or []
        ids = vector_data.get("ids") or []

        for i, document in enumerate(documents):
            vector_items.append({
                "id": ids[i] if i < len(ids) else "",
                "content": document,
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })
    except Exception as exc:
        vector_items = [{"id": "", "content": f"(vector store unavailable: {exc})", "metadata": {}}]

    sql_messages = load_recent_db_rows(limit=30)

    return jsonify({
        "vector_memories": _jsonable(vector_items),
        "recent_messages": _jsonable(sql_messages),
    })


@app.route("/api/message", methods=["POST"])
def send_message():
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
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"

    print("\nBabyClaw GUI is starting.")
    print(f"Workspace: {WORKSPACE.root}")
    print(f"Memory DB: {DB_PATH}")
    print(f"Vector DB: {VECTOR_DIR}")
    print(f"Opening {url} in your browser.\n")

    threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
