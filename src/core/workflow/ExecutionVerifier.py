from src.tools.utils import WorkspaceConfig
from src.action_constants import CONTENT_MUTATION_TOOLS

class ExecutionVerifier:
    """
        Deterministically verifies observable workspace state after execution.

        This is separate from the ReviewerAgent: it does not judge semantic
        quality, only whether mutating tools produced the filesystem state their
        resolved arguments claim they should have produced.
    """

    def __init__(self, workspace_config: WorkspaceConfig):
        self.workspace_config = workspace_config

    def verify(self, executor_response: dict) -> dict:
        issues = []
        execution_trace = executor_response.get("execution_trace", [])

        for step in execution_trace:
            if step.get("status") != "completed":
                continue

            tool_name = step.get("tool")
            resolved_args = step.get("resolved_args", {})

            try:
                issue = self.verify_step(
                    tool_name=tool_name,
                    resolved_args=resolved_args,
                    step=step,
                    execution_trace=execution_trace
                )
            except Exception as exc:
                issue = f"Execution verification failed for tool '{tool_name}': {exc}"

            if issue:
                issues.append(issue)

        if issues:
            return {
                "accepted": False,
                "review_summary": "Execution verifier found workspace state problems.",
                "issues": issues
            }

        return {
            "accepted": True,
            "review_summary": "Execution verifier confirmed observable workspace state.",
            "issues": []
        }

    def verify_step(self, tool_name: str, resolved_args: dict, step: dict = None,
                    execution_trace: list[dict] = None) -> str | None:
        if tool_name == "create_file":
            if self.has_later_content_mutation(step=step, execution_trace=execution_trace):
                return self.verify_file_exists_as_file(path=resolved_args.get("path"))

            return self.verify_file_exact_content(
                path=resolved_args.get("path"),
                content=resolved_args.get("content", ""),
                action="created"
            )

        if tool_name == "write_file":
            return self.verify_file_exact_content(
                path=resolved_args.get("path"),
                content=resolved_args.get("content", ""),
                action="written"
            )

        if tool_name == "append_file":
            return self.verify_file_contains_content(
                path=resolved_args.get("path"),
                content=resolved_args.get("content", ""),
                action="appended"
            )

        if tool_name == "delete_file":
            return self.verify_path_missing(path=resolved_args.get("path"), expected_kind="file")

        if tool_name == "create_dir":
            return self.verify_directory_exists(path=resolved_args.get("path"))

        if tool_name == "delete_dir":
            return self.verify_path_missing(path=resolved_args.get("path"), expected_kind="directory")

        if tool_name == "move_path":
            source_issue = self.verify_path_missing(path=resolved_args.get("source_path"), expected_kind="source path")
            if source_issue:
                return source_issue
            return self.verify_path_exists(path=resolved_args.get("destination_path"), expected_kind="destination path")

        if tool_name == "copy_path":
            source_issue = self.verify_path_exists(path=resolved_args.get("source_path"), expected_kind="source path")
            if source_issue:
                return source_issue
            return self.verify_path_exists(path=resolved_args.get("destination_path"), expected_kind="destination path")

        if tool_name == "replace_text":
            return self.verify_file_contains_content(
                path=resolved_args.get("path"),
                content=resolved_args.get("new_text", ""),
                action="updated"
            )

        return None

    def has_later_content_mutation(self, step: dict = None, execution_trace: list[dict] = None) -> bool:
        if not step or not execution_trace:
            return False

        path = (step.get("resolved_args") or {}).get("path")
        step_id = step.get("id")
        workflow_iteration = step.get("workflow_iteration", 1)

        if not path or step_id is None:
            return False

        current_key = (workflow_iteration, step_id)

        for later_step in execution_trace:
            if later_step.get("status") != "completed":
                continue

            later_key = (
                later_step.get("workflow_iteration", 1),
                later_step.get("id", 0)
            )

            if later_key <= current_key:
                continue

            if later_step.get("tool") not in CONTENT_MUTATION_TOOLS:
                continue

            later_path = (later_step.get("resolved_args") or {}).get("path")
            if later_path == path:
                return True

        return False

    def resolve_path(self, path: str):
        if not isinstance(path, str) or path.strip() == "":
            raise ValueError("missing path")

        return self.workspace_config.resolve_workspace_path(path)

    def verify_path_exists(self, path: str, expected_kind: str) -> str | None:
        target = self.resolve_path(path)

        if not target.exists():
            return f"Expected {expected_kind} '{path}' to exist after execution, but it does not."

        return None

    def verify_file_exists_as_file(self, path: str) -> str | None:
        target = self.resolve_path(path)

        if not target.exists():
            return f"Expected file '{path}' to exist after execution, but it does not."

        if not target.is_file():
            return f"Expected '{path}' to be a file after execution, but it is not."

        return None

    def verify_path_missing(self, path: str, expected_kind: str) -> str | None:
        target = self.resolve_path(path)

        if target.exists():
            return f"Expected {expected_kind} '{path}' to be removed after execution, but it still exists."

        return None

    def verify_directory_exists(self, path: str) -> str | None:
        target = self.resolve_path(path)

        if not target.exists():
            return f"Expected directory '{path}' to exist after execution, but it does not."

        if not target.is_dir():
            return f"Expected '{path}' to be a directory after execution, but it is not."

        return None

    def verify_file_exact_content(self, path: str, content: str, action: str) -> str | None:
        target = self.resolve_path(path)

        if not target.exists():
            return f"Expected file '{path}' to exist after execution, but it does not."

        if not target.is_file():
            return f"Expected '{path}' to be a file after execution, but it is not."

        actual_content = target.read_text(encoding="utf-8")
        expected_content = content or ""

        if actual_content != expected_content:
            return f"Expected file '{path}' to contain the {action} content, but final content differs."

        return None

    def verify_file_contains_content(self, path: str, content: str, action: str) -> str | None:
        target = self.resolve_path(path)

        if not target.exists():
            return f"Expected file '{path}' to exist after execution, but it does not."

        if not target.is_file():
            return f"Expected '{path}' to be a file after execution, but it is not."

        expected_content = content or ""

        if expected_content and expected_content not in target.read_text(encoding="utf-8"):
            return f"Expected file '{path}' to contain the {action} content, but it does not."

        return None
