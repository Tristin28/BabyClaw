from src.Agents.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Coordinator import Coordinator
from src.OllamaClient import OllamaClient
from src.Memory.sql_database import DatabaseManager
from src.Agents.MemoryAgent import MemoryAgent
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.tool_registry import build_tool_registry
from pathlib import Path

def main():
    llm = OllamaClient(model="qwen2.5:3b")
    db_path = Path(__file__).resolve().parents[1] / "babyclaw.db"
    db = DatabaseManager(db_path)
    db.init_db()
    planner = PlannerAgent(llm_client=llm)
    executor = ExecutorAgent(llm_client=llm, tool_registry=build_tool_registry(llm_client=llm))
    memory = MemoryAgent(db_manager=db, llm_client=llm)
    coordinator = Coordinator(planner=planner, executor=executor, reviewer=None, memory=memory,
                              planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS, tool_registry=build_tool_registry(llm_client=llm)
                              )


    conversation_id = 1

    print("\n---- FIRST RUN ----")
    result1 = coordinator.handle_user_request(conversation_id, "Summarise document.txt")
    print(result1)

    print("\n---- SECOND RUN ----")
    result2 = coordinator.handle_user_request(conversation_id, "Tell me what happened before")
    print(result2)

if __name__ == "__main__":
    main()