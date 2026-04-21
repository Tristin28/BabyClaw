from src.Agents.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Coordinator import Coordinator
from src.OllamaClient import OllamaClient
from src.message import Message

from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.tool_registry import build_tool_registry

def main():
    llm = OllamaClient(model="qwen2.5:3b")

    planner = PlannerAgent(llm_client=llm)
    executor = ExecutorAgent(llm_client=llm, tool_registry=build_tool_registry(llm_client=llm))

    coordinator = Coordinator(planner=planner, executor=executor, reviewer=None, memory=None,
                              planner_tool_descriptions=PLANNER_TOOL_DESCRIPTIONS, tool_registry=build_tool_registry(llm_client=llm)
                              )

    # Make sure workspace/document.txt exists before running this test
    user_task = "Read document.txt, and summarise it"

    result = coordinator.handle_user_request(conversation_id=1, user_task=user_task)

    print("\n---- FINAL COORDINATOR RESULT ----")
    print(result)

if __name__ == "__main__":
    main()