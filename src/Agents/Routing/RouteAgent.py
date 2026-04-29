from src.agents.BaseAgent import Agent
from src.llm.OllamaClient import OllamaClient
from src.core.message import Message
import re

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
                creative writing, drafting, rewriting, coding in chat, or answering questions.

                Creative writing, drafting, explaining, coding-in-chat, and normal
                questions are direct_response unless the current task explicitly
                refers to previous context or asks to save/write content into a file.

                Examples:
                - explain recursion
                - write me a letter
                - write a poem in chat
                - write a short story
                - write a poem about Jake
                - write bubble sort in Python
                - what is reinforcement learning?

                2. memory_question
                Use this when the user asks about remembered facts or preferences.

                Examples:
                - what is my name?
                - what do you remember about me?
                - what do you know about my project?

                3. contextual_followup
                Use this only when the current task contains explicit
                context-dependent language and cannot stand alone without previous
                conversation.

                Examples:
                - continue
                - continue it
                - do the same thing again
                - do the same for Jake
                - another one
                - write another poem
                - make it shorter
                - use the previous result
                - use that file/result
                - fix that
                - explain it further
                - write one about him/her/them when the referent only exists in previous conversation

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

                Also use this when the user asks to generate/write content:
                - in a text file
                - into a text file
                - in a file
                - as a .txt file
                - save it as a file
                - put it in a file
                Even if no exact filename is provided, this still means the user
                wants a workspace file created or written.

                Examples:
                - create test.txt with hello
                - write a short story in a text file
                - write a poem in a text file
                - save this as output.md
                - append hello to notes.txt
                - delete old.txt
                - rename a.txt to b.txt

                Important:
                - "write a letter" is direct_response unless the user asks to save it into a file.
                - "write code" is direct_response unless the user asks to create/save a file.
                - "write a poem in chat" is direct_response.
                - "write a short story" is direct_response.
                - "write a poem about Jake" is direct_response.
                - "write a short story in a text file" is workspace_mutation.
                - "write a poem in a text file" is workspace_mutation.
                - Do not classify creative writing as contextual_followup just because
                  previous conversation might contain useful inspiration.
                - Only classify contextual_followup when the current task explicitly
                  uses follow-up language such as continue, same, another, shorter,
                  improve it, fix that, previous result, that file/result, explain it
                  further, or unresolved pronouns like him/her/them.
                - Do not infer file operations unless the user clearly asks for a file/folder/path operation.

                Context is only used when the current user task explicitly depends on it.

                Return only valid JSON.
                """

    def __init__(self, llm_client: OllamaClient):
        super().__init__("router")
        self.llm_client = llm_client

    def has_explicit_followup_language(self, user_task: str) -> bool:
        task = " ".join(user_task.strip().lower().split())

        direct_phrases = [
            "continue",
            "do the same",
            "same thing",
            "another one",
            "another",
            "shorter",
            "longer",
            "explain it further",
            "explain that further",
            "explain this further",
            "write one about him",
            "write one about her",
            "write one about them",
            "about him",
            "about her",
            "about them",
        ]

        pronouns = ["it", "that", "this", "them", "those"]

        action_phrases = []
        action_phrases += [f"make {p}" for p in pronouns]
        action_phrases += [f"improve {p}" for p in pronouns]
        action_phrases += [f"fix {p}" for p in pronouns]

        previous_refs = [
            "use previous result",
            "use previous answer",
            "use previous response",
            "use previous one",
            "use the previous result",
            "use the previous answer",
            "use the previous response",
            "use the previous one",
        ]

        current_refs = [
            "use that file",
            "use that result",
            "use that answer",
            "use that response",
            "use that one",
            "use this file",
            "use this result",
            "use this answer",
            "use this response",
            "use this one",
        ]

        phrases = direct_phrases + action_phrases + previous_refs + current_refs

        return contains_any(task, phrases)


    def explicitly_requests_chat(self, user_task: str) -> bool:
        task = " ".join(user_task.strip().lower().split())

        chat_phrases = [
            "in chat",
            "in the chat",
            "in this chat",
            "here in chat",
            "display in chat",
            "display it in chat",
        ]

        return contains_any(task, chat_phrases)


    def explicitly_requests_file_output(self, user_task: str) -> bool:
        if self.explicitly_requests_chat(user_task):
            return False

        task = " ".join(user_task.strip().lower().split())

        file_output_phrases = [
            "create a text file",
            "create text file",
            "create a file",
            "create file",
            "in a text file",
            "into a text file",
            "in a file",
            "into a file",
            "as a .txt file",
            "as a txt file",
            "save it as a file",
            "save this as a file",
            "save that as a file",
            "save the result as a file",
            "save the output as a file",
            "put it in a file",
            "put this in a file",
            "put that in a file",
            "put the result in a file",
            "put the output in a file",
        ]

        if contains_any(task, file_output_phrases):
            return True

        file_action_patterns = [
            ("create", "file"),
            ("create", "text file"),
            ("write", "inside it"),
            ("write", "inside this"),
            ("write", "inside that"),
            ("write", "inside the file"),
            ("write", "inside a file"),
            ("write", "inside a text file"),
        ]

        return any(action in task and target in task for action, target in file_action_patterns)

    def normalise_response(self, user_task: str, response: dict) -> dict:
        """
            Guard against common router LLM drift.

            Contextual follow-up is only valid when the current task has explicit
            context-dependent language. Standalone creative/chat tasks must remain
            direct_response. Clear file-output phrasing must become
            workspace_mutation even when no exact filename is provided.
        """
        if not isinstance(response, dict):
            return response

        if self.explicitly_requests_file_output(user_task):
            normalised = dict(response)
            normalised["task_type"] = "workspace_mutation"
            normalised["routing_reason"] = (
                "The current task explicitly asks for generated content in a file; "
                "using workspace_mutation."
            )
            normalised["confidence"] = max(float(normalised.get("confidence", 0.0)), 0.95)
            return normalised

        if response.get("task_type") != "contextual_followup":
            return response

        if self.has_explicit_followup_language(user_task):
            return response

        normalised = dict(response)
        normalised["task_type"] = "direct_response"
        normalised["routing_reason"] = (
            "Standalone chat or creative-writing task without explicit follow-up language; "
            "using direct_response."
        )
        normalised["confidence"] = min(float(normalised.get("confidence", 1.0)), 0.9)
        return normalised

    def build_messages(self, user_task: str) -> list[dict]:
        return [
            {"role": "system", "content": RouteAgent.SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this user task:\n{user_task}"}
        ]

    def run(self, conversation_id: int, step_index: int, user_task: str) -> Message:
        try:
            messages = self.build_messages(user_task)
            response = self.llm_client.invoke_json(messages=messages, stream=False, schema=RouteAgent.SCHEMA)
            response = self.normalise_response(user_task=user_task, response=response)
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



def contains_any(task: str, phrases: list[str]) -> bool:
    return any(phrase in task for phrase in phrases)


def contains_any_pair(task: str, prefixes: list[str], targets: list[str]) -> bool:
    return any(f"{prefix} {target}" in task for prefix in prefixes for target in targets)
