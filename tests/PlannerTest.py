from src.Agents.PlannerAgent import PlannerAgent
from src.OllamaClient import OllamaClient


def main():
    llm = OllamaClient(model="qwen2.5:3b")
    planner = PlannerAgent(llm)

    planner_input = {
        "task": "Summarise the document.txt file",
        "context": "User often wants concise summaries.",
        "recent_messages": [
            {"sender": "user", "content": "Please summarise my file."}
        ],
        "tools": ["read_file", "summarise_text"],
        "conversation_id": 1,
        "step_index": 1,
    }

    result = planner.get_plan(planner_input)

    print("Planner returned:")
    print(result)

if __name__ == "__main__":
    main()