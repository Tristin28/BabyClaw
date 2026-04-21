from pathlib import Path
from src.Memory.database import DatabaseManager
from src.Memory.MessageRepository import MessageRepository
from src.message import Message


def main():
    db_path = Path(__file__).resolve().parents[1] / "babyclaw.db"

    db_manager = DatabaseManager(str(db_path))
    db_manager.init_db()

    repo = MessageRepository(db_manager)

    msg1 = Message(
        conversation_id=1,
        step_index=1,
        sender="user",
        receiver="coordinator",
        target_agent="planner",
        message_type="user_request",
        status="completed",
        response={"text": "Summarise document.txt"},
        visibility="external",
    )

    msg2 = Message(
        conversation_id=1,
        step_index=2,
        sender="planner",
        receiver="coordinator",
        target_agent="executor",
        message_type="plan",
        status="completed",
        response={
            "goal": "Summarise document.txt",
            "steps": [
                {"id": 1, "tool": "read_file", "args": {"path": "document.txt"}, "depends_on": []},
                {"id": 2, "tool": "summarise_txt", "args": {"source_step": 1}, "depends_on": [1]},
            ],
            "planning_rationale": "Need to read before summarising."
        },
        visibility="internal",
    )

    msg3 = Message(
        conversation_id=1,
        step_index=3,
        sender="executor",
        receiver="coordinator",
        target_agent="reviewer",
        message_type="execution_result",
        status="completed",
        response={
            "goal": "Summarise document.txt",
            "step_results": [
                {"id": 1, "tool": "read_file", "status": "completed", "result": "Some file contents"},
                {"id": 2, "tool": "summarise_txt", "status": "completed", "result": "Short summary"}
            ]
        },
        visibility="internal",
    )

    repo.store_message(msg1)
    repo.store_message(msg2)
    repo.store_message(msg3)

    recent = repo.get_recent_messages(conversation_id=1, k=2)

    print("\n---- LAST 2 MESSAGES ----")
    for msg in recent:
        print(msg)


    print("\n---- GETTING ALL MESSAGES ----")
    recent = repo.get_recent_messages(conversation_id=1, k=5)
    for msg in recent:
        print(msg)


    print("\n TESTING OUT INVALID ID")
    recent = repo.get_recent_messages(conversation_id=999, k=5)
    for msg in recent:
        print(msg)

if __name__ == "__main__":
    main()