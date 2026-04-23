from src.Agents.PlannerAgent import PlannerAgent
from src.OllamaClient import OllamaClient

def main():
    llm = OllamaClient(model="qwen2.5:3b")
    planner = PlannerAgent(llm)

    planner_input = {
        "task": "Summarise the document.txt file",
        "context": "User often wants concise summaries.",
        "k_recent_messages": [
            {"sender": "user", "content": "I have to study about the concepts within this file."}
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "Reads the contents of a text file",
                "args_schema": {
                    "file_id": "string"
                }
            },
            {
                "name": "summarise_txt",
                "description": "Summarises text content from a previous step",
                "args_schema": {
                    "source_step": "integer"
                }
            }
        ],
        "conversation_id": 1,
        "step_index": 1,
    }

    result = planner.run(planner_input)

    print("Planner returned:")
    print(result)

if __name__ == "__main__":
    main()