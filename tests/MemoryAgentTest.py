from pathlib import Path
from src.memory.sql_database import DatabaseManager
from src.agents.memory.MemoryAgent import MemoryAgent
from src.core.message import Message


def test_memory_prompt_uses_general_extraction_guide():
    agent = MemoryAgent.__new__(MemoryAgent)

    messages = agent.build_messages(
        "append to greetings.txt and say my surname which is Bezzina"
    )

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "GENERAL EXTRACTION GUIDE" in system_prompt
    assert "identity attributes: name, surname" in system_prompt
    assert '"my <attribute> which is <value>"' in system_prompt
    assert "relationship_<role>" in system_prompt
    assert "The user's surname is Bezzina." in system_prompt
    assert "ignore the task instruction" in system_prompt
    assert "append to greetings.txt and say my surname which is Bezzina" in user_prompt


def test_memory_validation_accepts_general_identity_fact():
    agent = MemoryAgent.__new__(MemoryAgent)

    memories = agent.validate_llm_response({
        "should_store": True,
        "memories": [
            {
                "memory_type": "user_fact",
                "topic": "identity_surname",
                "content": "The user's surname is Bezzina.",
                "confidence": 0.95,
            }
        ],
    })

    assert memories == [
        {
            "memory_type": "user_fact",
            "topic": "identity_surname",
            "content": "The user's surname is Bezzina.",
            "confidence": 0.95,
        }
    ]


def test_memory_rejection_keeps_task_details_out_but_allows_surname_fact():
    agent = MemoryAgent.__new__(MemoryAgent)

    assert not agent.should_reject_memory({
        "memory_type": "user_fact",
        "topic": "identity_surname",
        "content": "The user's surname is Bezzina.",
        "confidence": 0.95,
    })

    assert agent.should_reject_memory({
        "memory_type": "user_fact",
        "topic": "temporary_file",
        "content": "The user created a file named greetings.txt.",
        "confidence": 0.95,
    })


def main():
    db_path = Path(__file__).resolve().parents[1] / "babyclaw.db"

    db_manager = DatabaseManager(str(db_path))
    db_manager.init_db()

    agent = MemoryAgent(db_manager)

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

    agent.store_message(msg1)
    agent.store_message(msg2)
    agent.store_message(msg3)

    recent = agent.get_recent_messages(conversation_id=1, k=2)

    print("\n---- LAST 2 MESSAGES ----")
    for msg in recent:
        print(msg)


    print("\n---- GETTING ALL MESSAGES ----")
    recent = agent.get_recent_messages(conversation_id=1, k=5)
    for msg in recent:
        print(msg)


    print("\n TESTING OUT INVALID ID")
    recent = agent.get_recent_messages(conversation_id=999, k=5)
    for msg in recent:
        print(msg)

if __name__ == "__main__":
    main()
