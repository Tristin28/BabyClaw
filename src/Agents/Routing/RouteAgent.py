from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message

class RouteAgent(Agent):
    SCHEMA = {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": ["direct_response", "memory_question", "contextual_followup",
                         "workspace_read", "workspace_summarise", "workspace_mutation"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "routing_reason": {
                "type": "string"
            }
        },
        "required": ["task_type", "confidence", "routing_reason"],
        "additionalProperties": False
    }


    SYSTEM_PROMPT = """
                You are the Routing Agent.

                Your only job is to classify the user's CURRENT TASK.

                You do not answer the user.
                You do not plan.
                You do not choose tools.
                You do not decide permissions.
                You only classify the task type.

                Choose exactly one task_type:

                1. direct_response
                Use this for normal chat, explanations, advice, writing text in chat,
                rewriting, coding in chat, or answering questions.

                Examples:
                - explain recursion
                - write me a letter
                - write bubble sort in Python
                - improve this paragraph
                - what is reinforcement learning?

                2. memory_question
                Use this when the user asks about remembered facts or preferences.

                Examples:
                - what is my name?
                - what do you remember about me?
                - what do you know about my project?

                3. contextual_followup
                Use this when the task clearly depends on previous conversation.

                Examples:
                - continue
                - do the same thing again
                - use the previous result
                - fix that
                - explain it further

                4. workspace_read
                Use this when the user wants to read, show, find, list, or search files/folders.

                Examples:
                - read notes.txt
                - list the files
                - find hello.txt
                - search for TODO

                5. workspace_summarise
                Use this when the user wants to summarise, explain, shorten, or describe an existing file.

                Examples:
                - summarise notes.txt
                - explain README.md
                - shorten this file

                6. workspace_mutation
                Use this when the user asks to create, save, write, append, edit, replace,
                delete, move, copy, or rename files/folders.

                Examples:
                - create test.txt with hello
                - save this as output.md
                - append hello to notes.txt
                - delete old.txt
                - rename a.txt to b.txt

                Important:
                - "write a letter" is direct_response unless the user asks to save it into a file.
                - "write code" is direct_response unless the user asks to create/save a file.
                - Do not infer file operations unless the user clearly asks for a file/folder/path operation.

                Return only valid JSON.
                """

    def __init__(self, llm_client: OllamaClient):
        super().__init__("router")
        self.llm_client = llm_client

    def build_messages(self, user_task: str) -> list[dict]:
        return [
            {"role": "system", "content": RouteAgent.SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this user task:\n{user_task}"}
        ]

    def run(self, conversation_id: int, step_index: int, user_task: str) -> Message:
        try:
            messages = self.build_messages(user_task)
            response = self.llm_client.invoke_json(messages=messages, stream=False, schema=RouteAgent.SCHEMA)
            status = "completed"
        except Exception as e:
            response = {
                "error": str(e),
                "task_type": "direct_response",
                "confidence": 0.0,
                "routing_reason": "Router failed, so safe fallback should be used."
            }
            status = "failed"


        return self.get_message(conversation_id=conversation_id, step_index=step_index,
                                receiver="coordinator", target_agent=None,
                                message_type="route", status=status,
                                response=response, visibility="internal")