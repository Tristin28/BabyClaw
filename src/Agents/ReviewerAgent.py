from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message

class ReviewerAgent(Agent):
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
                "content": """
                            You are a reviewer agent.

                            Your job is to inspect the executor result and decide whether it satisfactorily fulfills the user's task.

                            Rules:
                            1. Judge whether the executor result actually addresses the user's task.
                            2. Accept only if the result is relevant, complete enough for the task, and not clearly incorrect.
                            3. Reject if the result is missing key parts, irrelevant, malformed, too vague, or clearly inconsistent with the task.
                            4. Reject results that only look like a valid answer format but do not match the actual subject or intent of the task.
                            5. A result must be semantically aligned with the task, not just be a short well-formed sentence.
                            6. Be strict but practical. Minor wording differences are acceptable only if the real task is still fulfilled.
                            7. Return only valid JSON matching the provided schema.
                            8. 'accepted' must be true only if the result is genuinely good enough to proceed.
                            9. 'review_summary' must be short and clear.
                            10. 'issues' must list concrete problems when rejected, and should be [] when accepted.
                            """
            },
            {
                "role": "user",
                "content": f"""
                            User task:
                            {user_task}

                            Execution results:
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