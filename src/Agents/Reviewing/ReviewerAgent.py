from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message
from src.Agents.Reviewing.ReviewPrompt import REVIEWER_SYSTEM_PROMPT
import json 

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
            
    def build_review_evidence(self, executor_response: dict, workspace_before: list = None, workspace_after: list = None) -> dict:
        """
            Builds the small clean object that the reviewer LLM is allowed to see.
            This hides architecture-only data such as rollback logs, approved actions,
            permission state, and internal execution state.
            Now also includes workspace listings before/after so the reviewer can
            verify negative claims like "deleted all files except X".
        """
        clean_steps = []

        for step in executor_response.get("execution_trace", []):
            clean_step = {
                "id": step.get("id"),
                "tool": step.get("tool"),
                "status": step.get("status"),
                "args": step.get("args", {}),
            }

            if "resolved_args" in step:
                clean_step["resolved_args"] = step["resolved_args"]

            if "result" in step:
                clean_step["result"] = step["result"]

            if "error" in step:
                clean_step["error"] = step["error"]

            clean_steps.append(clean_step)

        return {
            "goal": executor_response.get("goal", ""),
            "steps": clean_steps,
            "workspace_before": workspace_before or [],
            "workspace_after": workspace_after or [],
        }

    def build_messages(self, user_task: str, review_evidence: dict) -> list[dict]:
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
                            Only judge whether the clean execution evidence satisfies the CURRENT USER TASK above.
                            Do not judge old user messages, memory, rollback data, permission data, or unrelated prior tasks.

                            Use 'workspace_before' and 'workspace_after' to verify negative or set-based claims
                            such as "delete all files except X" or "remove every .txt file". If workspace_after
                            matches what the task requested, ACCEPT, regardless of how many delete steps were used.

                            CLEAN EXECUTION EVIDENCE:
                            {json.dumps(review_evidence, indent=2)}
                            """
            }
        ]

    def run(self, conversation_id: int, step_index: int, user_task: str, execution_response: dict, workspace_before: list = None, workspace_after: list = None) -> Message:
        try:
            self.validate_input(user_task=user_task, execution_trace=execution_response, conversation_id=conversation_id, step_index=step_index)

            review_evidence = self.build_review_evidence(execution_response, workspace_before=workspace_before, workspace_after=workspace_after)
            messages = self.build_messages(user_task=user_task, review_evidence=review_evidence)

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