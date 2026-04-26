from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message
from src.Agents.Reviewing.ReviewPrompt import REVIEWER_SYSTEM_PROMPT

class ReviewerAgent(Agent):
    REVIEWER_SYSTEM_PROMPT = REVIEWER_SYSTEM_PROMPT
    SCHEMA = {
        "type": "object",
        "properties": {
            "accepted": {
                "type": "boolean",
                "description": "Whether the executor result satisfactorily fulfills the user task."
            },
            "review_summary": {
                "type": "string",
                "description": "Short explanation of the review decision."
            },
            "issues": {
                "type": "array",
                "description": "List of problems found in the executor result. Empty if accepted.",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["accepted", "review_summary", "issues"]
    }

    def __init__(self, llm_client: OllamaClient):
        super().__init__("reviewer")
        self.llm_client = llm_client

    def validate_input(self, user_task: str, execution_trace: dict, conversation_id: int, step_index: int):
        if not isinstance(user_task, str) or user_task.strip() == "":
            raise ValueError("Reviewer input 'user_task' must be a non-empty string")

        if not isinstance(execution_trace, dict):
            raise ValueError("Reviewer input 'execution_trace' must be a dictionary")

        if not isinstance(conversation_id, int):
            raise ValueError("Reviewer input 'conversation_id' must be an integer")

        if not isinstance(step_index, int):
            raise ValueError("Reviewer input 'step_index' must be an integer")

    def validate_llm_response(self, review_response: dict):
        if not isinstance(review_response, dict):
            raise ValueError("Reviewer response must be a dictionary")

        required_fields = {"accepted": bool, "review_summary": str, "issues": list}

        for field, expected_type in required_fields.items():
            if field not in review_response:
                raise ValueError(f"Reviewer response missing '{field}'")

            if not isinstance(review_response[field], expected_type):
                raise ValueError(f"Reviewer response field '{field}' must be {expected_type.__name__}")

        for issue in review_response["issues"]:
            if not isinstance(issue, str):
                raise ValueError("Each issue must be a string")

    def build_messages(self, user_task: str, execution_trace: dict) -> list[dict]:
        return [
            {
                "role": "system",
                "content": ReviewerAgent.REVIEWER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                            CURRENT USER TASK TO REVIEW:
                            {user_task}

                            Important:
                            Only judge whether the execution results satisfy the CURRENT USER TASK above.
                            Do not judge old user messages, recent conversation, memory, or unrelated prior tasks.

                            EXECUTION RESULTS:
                            {execution_trace}
                            """
            }
        ]

    def run(self, conversation_id: int, step_index: int, user_task: str, execution_trace: dict) -> Message:
        try:
            self.validate_input(user_task=user_task, execution_trace=execution_trace, conversation_id=conversation_id, step_index=step_index)

            messages = self.build_messages(user_task=user_task, execution_trace=execution_trace)

            review_response = self.llm_client.invoke_json(messages=messages, stream=False, schema=self.SCHEMA)
            self.validate_llm_response(review_response)

            status = "completed"
            target_agent = None

        except Exception as e:
            review_response = {
                "accepted": False,
                "review_summary": "Reviewer failed to produce a valid review.",
                "issues": [str(e)]
            }
            status = "failed"
            target_agent = None

        return self.get_message(conversation_id=conversation_id, step_index=step_index, receiver="coordinator", target_agent=target_agent, message_type="review_result",
                                status=status, response=review_response, visibility="internal")