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

                        Return only valid JSON.

                        ==================================================
                        TASK TYPES

                        Choose exactly one task_type:

                        1. direct_response
                        Use for normal chat, explanations, advice, drafting text in chat, creative
                        writing in chat, code in chat, and answering general questions.

                        Examples:
                        - "what is reinforcement learning?" -> direct_response
                        - "explain recursion" -> direct_response
                        - "write a short poem about a robot" -> direct_response
                        - "draft a polite email" -> direct_response
                        - "write bubble sort in Python" -> direct_response

                        2. memory_question
                        Use when the user asks about remembered user facts or preferences.

                        Examples:
                        - "what is my name?" -> memory_question
                        - "what do you remember about me?" -> memory_question
                        - "what do you know about my project?" -> memory_question

                        3. contextual_followup
                        Use only when the current task explicitly depends on previous conversation
                        or a previously generated answer/file. The task on its own would not be
                        understandable.

                        Examples:
                        - "make it shorter" -> contextual_followup
                        - "continue from the previous answer" -> contextual_followup
                        - "do the same for Jake" -> contextual_followup
                        - "write another version" -> contextual_followup
                        - "use the previous result" -> contextual_followup
                        - "explain that further" -> contextual_followup

                        Do NOT use contextual_followup for standalone tasks that just happen to
                        contain a pronoun. "What is it used for?" with no obvious referent is a
                        direct_response, not a follow-up.

                        4. workspace_read
                        Use when the user wants to read, show, open, view, list, find, or search
                        files/folders in the workspace.

                        Examples:
                        - "read notes.txt" -> workspace_read
                        - "show me Coordinator.py" -> workspace_read
                        - "list the files" -> workspace_read
                        - "find hello.txt" -> workspace_read
                        - "search for TODO" -> workspace_read

                        5. workspace_summarise
                        Use when the user wants a summary, explanation, or short description of an
                        existing file in the workspace.

                        Examples:
                        - "summarise notes.txt" -> workspace_summarise
                        - "explain README.md" -> workspace_summarise
                        - "shorten this file" -> workspace_summarise

                        6. workspace_mutation
                        Use when the user asks to create, save, write, append, edit, replace,
                        delete, move, copy, or rename files/folders, or to save generated content
                        into a file.

                        Examples:
                        - "create a file called notes.txt" -> workspace_mutation
                        - "write this into summary.txt" -> workspace_mutation
                        - "save it as output.md" -> workspace_mutation
                        - "append hello to notes.txt" -> workspace_mutation
                        - "delete old.txt" -> workspace_mutation
                        - "rename a.txt to b.txt" -> workspace_mutation
                        - "write a poem in a text file" -> workspace_mutation
                        - "create a text file and put my notes inside" -> workspace_mutation

                        ==================================================
                        IMPORTANT RULES

                        - Creative writing in chat is direct_response. Only switch to
                        workspace_mutation if the user clearly asks for a file/folder to be
                        created or written.
                        - If the user explicitly says "in chat" or "display in chat", the answer
                        belongs in chat, not a file.
                        - Do not infer file operations from generic verbs ("write", "make")
                        unless the task clearly mentions a file, folder, path, extension, or
                        saving.
                        - contextual_followup is for tasks that genuinely need previous turns to
                        make sense. Do not pick it just because a pronoun appears.

                        Return only valid JSON matching the schema.
                    """

    def __init__(self, llm_client: OllamaClient):
        super().__init__("router")
        self.llm_client = llm_client

    def explicitly_requests_chat(self, user_task: str) -> bool:
        """
        Conservative safety guard: when the user explicitly asks for the
        answer in chat, the LLM should never route the task to a file
        mutation.
        """
        task = " ".join(user_task.strip().lower().split())

        chat_phrases = [
            "in chat",
            "in the chat",
            "in this chat",
            "here in chat",
            "display in chat",
            "display it in chat",
        ]

        return any(phrase in task for phrase in chat_phrases)
    
    def explicitly_requests_workspace_mutation(self, user_task: str) -> bool:
        """
        Workspace mutation is a side effect, so the router LLM must provide a
        classification that is grounded in explicit user text. This guard does
        not try to enumerate every possible chat/drafting request. It only asks
        whether the user clearly requested a file/folder/path/workspace effect.
        """
        if not isinstance(user_task, str):
            return False

        task = " ".join(user_task.strip().lower().split())

        if not task:
            return False

        mutation_action = re.search(
            r"\b(?:save|create|write|append|edit|delete|remove|move|copy|rename|overwrite|replace|put)\b",
            task
        )

        if not mutation_action:
            return False

        filename_or_path = re.search(
            r"(?<![\w./-])[\w./-]+\.[a-z0-9]{1,12}(?![\w./-])",
            task
        )

        workspace_target = re.search(
            r"\b(?:file|folder|directory|path|workspace)\b",
            task
        )

        destination_phrase = re.search(
            r"\b(?:save|write|append|put)\b.+\b(?:to|into|inside|in)\b",
            task
        )

        destructive_or_structural_action = re.search(
            r"\b(?:delete|remove|move|copy|rename|overwrite|replace|edit)\b",
            task
        )

        return bool(
            filename_or_path
            or workspace_target
            or (destination_phrase and (filename_or_path or workspace_target))
            or (destructive_or_structural_action and (filename_or_path or workspace_target))
        )

    def normalise_response(self, user_task: str, response: dict) -> dict:
        """
        Treat the LLM classification as a proposal.

        If the user explicitly asks for chat, refuse routes that would write to
        the workspace. If the LLM proposes workspace_mutation, require explicit
        side-effect evidence from the original user task before allowing it.
        """
        if not isinstance(response, dict):
            return response

        if response.get("task_type") == "direct_response":
            return response

        if self.explicitly_requests_chat(user_task):
            normalised = dict(response)
            normalised["task_type"] = "direct_response"
            normalised["routing_reason"] = (
                "User explicitly asked for the answer in chat; forcing direct_response."
            )
            normalised["confidence"] = max(float(normalised.get("confidence", 0.0)), 1.0)
            return normalised

        if response.get("task_type") == "workspace_mutation" and not self.explicitly_requests_workspace_mutation(user_task):
            normalised = dict(response)
            normalised["task_type"] = "direct_response"
            normalised["routing_reason"] = (
                "Router proposed workspace_mutation, but the user did not explicitly "
                "request a file, folder, path, workspace, filename, or workspace side effect."
            )
            normalised["confidence"] = max(float(normalised.get("confidence", 0.0)), 1.0)
            return normalised

        return response

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
