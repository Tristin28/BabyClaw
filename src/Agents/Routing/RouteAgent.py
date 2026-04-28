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
            "tool_group": {
                "type": "string",
                "enum": ["direct_response_tools", "read_file_tools",
                         "summarise_file_tools", "mutation_file_tools"]
            },
            "memory_mode": {
                "type": "string",
                "enum": ["none", "pinned_only", "relevant_only", "full"]
            },
            "use_recent_messages": {"type": "boolean"},
            "use_workspace": {"type": "boolean"},
            "routing_reason": {"type": "string"},
        },
        "required": ["task_type", "tool_group", "memory_mode",
                     "use_recent_messages", "use_workspace", "routing_reason"],
        "additionalProperties": False,
    }

    SYSTEM_PROMPT = """
            You are a deterministic ROUTING AGENT. Your only job is to classify the current user task.
            You do NOT plan, execute, or answer the user. You output ONLY structured JSON.

            Choose ONE task_type:

            - direct_response: chat, explanations, advice, drafts, code in chat, writing letters/emails IN CHAT.
            Examples: "explain recursion", "write me a letter", "write bubble sort code", "draft an email to John"

            - memory_question: user asks about stored personal facts/preferences.
            Examples: "what is my name?", "what do you remember about me?"

            - contextual_followup: user clearly refers to prior conversation.
            Examples: "continue", "same as before", "use that result", "do it again"

            - workspace_read: read/show/open/list/find/search workspace files/folders.
            Examples: "read notes.txt", "list files", "find hello.txt"

            - workspace_summarise: summarise/explain/shorten an existing workspace file.
            Examples: "summarise notes.txt", "explain README.md"

            - workspace_mutation: create/write/save/append/edit/replace/delete/move/copy/rename files/folders.
            Examples: "create notes.txt", "save this into output.md", "delete test.py"

            CRITICAL RULES:
            - "write a letter" alone is direct_response. Only workspace_mutation if the user names a file
            (e.g. "save the letter as letter.txt").
            - Do not infer file operations from chat tasks.

            tool_group must match task_type:
            - direct_response, memory_question, contextual_followup -> direct_response_tools
            - workspace_read -> read_file_tools
            - workspace_summarise -> summarise_file_tools
            - workspace_mutation -> mutation_file_tools

            memory_mode:
            - none: workspace_read, workspace_summarise, workspace_mutation
            - pinned_only: direct_response, memory_question
            - full: contextual_followup
            - relevant_only: rare, only when user explicitly references project history

            use_recent_messages: true ONLY for contextual_followup. Otherwise false.
            use_workspace: true ONLY for workspace_read/summarise/mutation. Otherwise false.

            routing_reason: one short sentence justifying the classification.
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
            response = {"error": str(e)}
            status = "failed"

        return self.get_message(conversation_id=conversation_id, step_index=step_index,
                                receiver="coordinator", target_agent=None,
                                message_type="route", status=status,
                                response=response, visibility="internal")