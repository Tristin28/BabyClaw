'''
    ActiveContext is a small explicit session-state record kept by the Coordinator and consumed by ContextResolver. It exists so that vague references such as
    "it", "the previous answer", "the file" can be resolved against concrete state instead of relying on the LLM to guess and potentially have more LLMs hallucinate

    The Coordinator owns ActiveContext Hence why it would be found composed (i.e. composition relationship) inside it. 
    It is updated:
        - when the user sends a new message,
        - after a successful workflow finishes (assistant response, generated content,created/modified files).
'''


class ActiveContext:
    def __init__(self):
        self.last_user_message: str | None = None
        self.last_assistant_response: str | None = None
        self.last_generated_content: str | None = None
        self.last_tool_result: str | None = None
        self.last_viewed_file: str | None = None
        self.last_modified_file: str | None = None
        self.last_created_file: str | None = None
        self.active_file: str | None = None
        self.last_workspace_path: str | None = None
        self.last_successful_action: str | None = None

    def record_user_message(self, user_task: str) -> None:
        self.last_user_message = user_task

    def record_assistant_response(self, content: str) -> None:
        if isinstance(content, str) and content.strip():
            self.last_assistant_response = content

    def record_generated_content(self, content: str) -> None:
        if isinstance(content, str) and content.strip():
            self.last_generated_content = content

    def record_tool_result(self, result) -> None:
        if isinstance(result, str) and result.strip():
            self.last_tool_result = result

    def record_file_viewed(self, path: str) -> None:
        if isinstance(path, str) and path.strip():
            self.last_viewed_file = path
            self.active_file = path

    def record_file_modified(self, path: str) -> None:
        if isinstance(path, str) and path.strip():
            self.last_modified_file = path
            self.active_file = path

    def record_file_created(self, path: str) -> None:
        if isinstance(path, str) and path.strip():
            self.last_created_file = path
            self.active_file = path

    def record_successful_action(self, description: str) -> None:
        if isinstance(description, str) and description.strip():
            self.last_successful_action = description

    def update_from_execution(self, executor_response: dict) -> None:
        '''
            Pulls the relevant signals out of a successful execution trace so ContextResolver can later resolve references like "it" or 
            "the previous file" without re-running the LLM.

            The mapping is intentionally narrow: only tools whose semantics are clear update state.
                (read/list/find -> viewed; create -> created; write/append/replace -> modified; delete -> active_file cleared) 
        '''
        if not isinstance(executor_response, dict):
            return

        trace = executor_response.get("execution_trace", []) or []

        for step in trace:
            if step.get("status") != "completed":
                continue

            tool_name = step.get("tool")
            args = step.get("resolved_args") or step.get("args", {}) or {}
            result = step.get("result")
            path = args.get("path") if isinstance(args, dict) else None

            if tool_name == "direct_response" and isinstance(result, str):
                self.record_assistant_response(result)

            if tool_name == "generate_content" and isinstance(result, str):
                self.record_generated_content(result)

            if tool_name in {"read_file", "summarise_txt"} and isinstance(path, str):
                self.record_file_viewed(path)

            if tool_name in {"find_file", "find_file_recursive", "list_dir", "list_tree", "search_text"} and isinstance(result, str):
                self.record_tool_result(result)

            if tool_name == "create_file" and isinstance(path, str):
                self.record_file_created(path)

            if tool_name in {"write_file", "append_file", "replace_text"} and isinstance(path, str):
                self.record_file_modified(path)

            if tool_name == "delete_file" and isinstance(path, str):
                if self.active_file == path:
                    self.active_file = None

    def snapshot(self) -> dict:
        return {
            "last_user_message": self.last_user_message,
            "last_assistant_response": self.last_assistant_response,
            "last_generated_content": self.last_generated_content,
            "last_tool_result": self.last_tool_result,
            "last_viewed_file": self.last_viewed_file,
            "last_modified_file": self.last_modified_file,
            "last_created_file": self.last_created_file,
            "active_file": self.active_file,
            "last_workspace_path": self.last_workspace_path,
            "last_successful_action": self.last_successful_action,
        }
