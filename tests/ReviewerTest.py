from src.Agents.ReviewerAgent import ReviewerAgent
from src.OllamaClient import OllamaClient

llm_client = OllamaClient(model="qwen2.5:3b")
reviewer = ReviewerAgent(llm_client)

def run_test(name, user_task, execution_trace):
    print(f"\n--- {name} ---")
    msg = reviewer.run(
        conversation_id=1,
        step_index=1,
        user_task=user_task,
        execution_trace=execution_trace
    )
    print("Status:", msg.status)
    print("Response:", msg.response)

# Test 1: good result
run_test(
    "GOOD RESULT",
    "Summarise the document in short terms",
    {
        "goal": "Summarise the document in short terms",
        "step_results": [
            {
                "id": 1,
                "tool": "summarise_tool",
                "status": "completed",
                "result": "The document explains gradient descent as a method that updates parameters step by step to reduce error."
            }
        ]
    }
)

# Test 2: incomplete result
run_test(
    "INCOMPLETE RESULT",
    "Summarise the document in short terms",
    {
        "goal": "Summarise the document in short terms",
        "step_results": [
            {
                "id": 1,
                "tool": "summarise_tool",
                "status": "completed",
                "result": ""
            }
        ]
    }
)

# Test 3: irrelevant result
run_test(
    "IRRELEVANT RESULT",
    "Summarise the document in short terms",
    {
        "goal": "Summarise the document in short terms",
        "step_results": [
            {
                "id": 1,
                "tool": "summarise_tool",
                "status": "completed",
                "result": "The weather is sunny today."
            }
        ]
    }
)