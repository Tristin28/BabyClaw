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
        "required": ["accepted", "review_summary", "issues"],
        "additionalProperties": False
    }

    MUTATION_TOOLS = {
        "create_file",
        "write_file",
        "append_file",
        "delete_file",
        "replace_text",
        "create_dir",
        "delete_dir",
        "move_path",
        "copy_path"
    }

    CONTENT_WRITING_TOOLS = {
        "create_file",
        "write_file",
        "append_file"
    }

    def __init__(self, llm_client: OllamaClient, workspace_config=None):
        super().__init__("reviewer")
        self.llm_client = llm_client
        self.workspace_config = workspace_config

    def validate_input(self, user_task: str, execution_trace: dict, conversation_id: int, step_index: int, route: dict = None):
        if not isinstance(user_task, str) or user_task.strip() == "":
            raise ValueError("Reviewer input 'user_task' must be a non-empty string")

        if not isinstance(execution_trace, dict):
            raise ValueError("Reviewer input 'execution_trace' must be a dictionary")

        if route is not None and not isinstance(route, dict):
            raise ValueError("Reviewer input 'route' must be a dictionary")

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

        # If reviewer accepts the workflow, there should not be any issues left.
        if review_response["accepted"] and review_response["issues"]:
            raise ValueError("Reviewer accepted the result but still returned issues.")

        # If reviewer rejects the workflow, it should explain why.
        if not review_response["accepted"] and not review_response["issues"]:
            raise ValueError("Reviewer rejected the result but did not return any issues.")

    def build_clean_route(self, route: dict = None) -> dict:
        """
            Builds the small clean route object that the reviewer LLM is allowed to see.
            This lets the reviewer check whether the executor stayed inside the scope
            which the coordinator allowed for this current task.
        """
        route = route or {}

        return {
            "task_type": route.get("task_type"),
            "tool_group": route.get("tool_group"),
            "use_workspace": route.get("use_workspace"),
            "allow_mutations": route.get("allow_mutations"),
            "allowed_tools": route.get("allowed_tools", []),
            "routing_reason": route.get("routing_reason", "")
        }

    def read_final_file_content(self, path: str) -> dict:
        """
            Best-effort final content evidence for generated/written files.
            Failure to read is returned as evidence instead of failing review.
        """
        if self.workspace_config is None:
            return {"available": False, "reason": "Reviewer has no workspace_config."}

        if not isinstance(path, str) or path.strip() == "":
            return {"available": False, "reason": "No valid path was available."}

        try:
            file_path = self.workspace_config.resolve_workspace_path(path)

            if not file_path.exists():
                return {"available": False, "reason": "File does not exist in final workspace state."}

            if not file_path.is_file():
                return {"available": False, "reason": "Path is not a file in final workspace state."}

            return {
                "available": True,
                "path": path,
                "content": file_path.read_text(encoding="utf-8")
            }
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def build_review_evidence(self, executor_response: dict, workspace_before: list = None, workspace_after: list = None, route: dict = None) -> dict:
        """
            Builds the small clean object that the reviewer LLM is allowed to see.
            This hides architecture-only data such as rollback logs, approved actions,
            permission state, and internal execution state.

            It includes workspace_before, workspace_after, and workspace_diff so the
            reviewer can check whether the final workspace state actually matches
            what the user asked for.
        """
        clean_steps = []

        for step in executor_response.get("execution_trace", []):
            tool_name = step.get("tool")
            resolved_args = step.get("resolved_args", {})

            clean_step = {
                "id": step.get("id"),
                "tool": tool_name,
                "status": step.get("status"),
                "args": step.get("args", {}),
            }

            if "resolved_args" in step:
                clean_step["resolved_args"] = resolved_args

            if "result" in step:
                clean_step["result"] = step["result"]

            if "error" in step:
                clean_step["error"] = step["error"]

            if tool_name in self.CONTENT_WRITING_TOOLS:
                clean_step["written_content"] = resolved_args.get("content", "")
                clean_step["final_content"] = self.read_final_file_content(resolved_args.get("path"))

            clean_steps.append(clean_step)

        return {
            "goal": executor_response.get("goal", ""),
            "route_scope": self.build_clean_route(route=route),
            "mutation_tools": sorted(self.MUTATION_TOOLS),
            "steps": clean_steps,
            "workspace_before": workspace_before or [],
            "workspace_after": workspace_after or [],
            "workspace_diff": self.build_workspace_diff(
                workspace_before=workspace_before,
                workspace_after=workspace_after
            )
        }
    
    def build_workspace_diff(self, workspace_before: list = None, workspace_after: list = None) -> dict:
        """
            Builds a simple before/after difference of the workspace.
            This helps the reviewer check what actually changed, instead of only
            trusting the execution trace.
        """
        before_set = set(workspace_before or [])
        after_set = set(workspace_after or [])

        created_paths = sorted(after_set - before_set)
        deleted_paths = sorted(before_set - after_set)
        unchanged_paths = sorted(before_set & after_set)

        return {
            "created_paths": created_paths,
            "deleted_paths": deleted_paths,
            "unchanged_paths": unchanged_paths
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

                            Use this priority order:

                            1. CURRENT USER TASK
                            This is the source of truth.

                            2. WORKSPACE DIFF
                            Check what paths were actually created, deleted, or left unchanged.

                            3. EXECUTED STEPS
                            Check the actual tools, resolved_args, results, and errors.

                            4. ROUTE SCOPE
                            Check whether the execution stayed inside the coordinator-approved route.

                            For workspace mutation tasks:
                            - Accept only if the final workspace state matches what the user asked for.
                            - Reject if the task asked to create a folder but no folder was created.
                            - Reject if the task asked to create files but no files were created.
                            - Reject if the created names are generic placeholders and not grounded in the task.
                            - Reject if file content is empty when the user asked for meaningful content.
                            - Reject if file content is unrelated to the requested topic.
                            - Reject if only generate_content ran but nothing was saved into the workspace.
                            - Reject if extra files/folders were created that the user did not ask for.
                            - Reject if requested files/folders are missing from workspace_after.
                            - Reject if the workspace changed even though the task was read-only.

                            For topic-specific creation tasks:
                            - Reject if the created content does not clearly relate to the requested topic.
                            - Reject if the file is only a shallow scaffold, placeholder, vague future intention, unrelated content, or generic file.
                            - Reject if the reviewer summary would need to describe the result using names/content that came from execution instead of the user task.
                            - The review must compare the execution against the CURRENT USER TASK, not against the planner goal.
                            - Do not accept only because a file/folder was created.
                            - If the user asks for a program, game, app, pipeline, algorithm, script, document, project, or system, the created content must be a meaningful minimal version of that requested artifact.

                            For file content:
                            - If the task says a file should contain specific content, the content must appear in resolved_args.content or the final file state.
                            - If the task asks for a topic-specific file, the content must be about that topic.
                            - Do not accept generic content like "Hello world", "Project Name", "sample text", or placeholder text unless the user asked for it.
                            - When final_content is present, judge the final saved content rather than tool success messages.

                            For route scope:
                            - If route_scope.allow_mutations is false, mutation tools should not have been used.
                            - If route_scope.allowed_tools is not empty, every executed tool must be inside allowed_tools.

                            Mutation tools are listed in 'mutation_tools'.

                            Before deciding, internally check:
                            - What did the user request?
                            - What did the executor actually create, change, read, or answer?
                            - What evidence in resolved_args, written_content, final_content, results, workspace_before, workspace_after, and workspace_diff proves the output satisfies the request?
                            - What is missing, unsupported, generic, or unrelated?

                            If accepted is true, issues must be [].
                            If accepted is false, issues must clearly explain what was missing, wrong, extra, or out of scope.

                            CLEAN EXECUTION EVIDENCE:
                            {json.dumps(review_evidence, indent=2)}
                            """
            }
        ]
    
    def deterministic_workspace_checks(self, user_task: str, review_evidence: dict) -> list[str]:
        """
            Fast deterministic checks before asking the LLM, these do not replace the LLM reviewer however they catch obvious workspace failures.
        """
        issues = []

        route_scope = review_evidence.get("route_scope", {})
        steps = review_evidence.get("steps", [])
        workspace_diff = review_evidence.get("workspace_diff", {})

        task_type = route_scope.get("task_type")
        allow_mutations = route_scope.get("allow_mutations")
        allowed_tools = set(route_scope.get("allowed_tools", []))

        created_paths = workspace_diff.get("created_paths", [])
        deleted_paths = workspace_diff.get("deleted_paths", [])

        executed_tools = [step.get("tool") for step in steps if step.get("status") == "completed"]

        if allowed_tools:
            for tool_name in executed_tools:
                if tool_name not in allowed_tools:
                    issues.append(f"Tool '{tool_name}' was executed outside the allowed route scope.")

        if allow_mutations is False:
            for tool_name in executed_tools:
                if tool_name in self.MUTATION_TOOLS:
                    issues.append(f"Mutation tool '{tool_name}' was used even though mutations were not allowed.")

        if task_type == "workspace_mutation":
            real_mutation_tools = self.MUTATION_TOOLS
            used_real_mutation = any(tool in real_mutation_tools for tool in executed_tools)

            if not used_real_mutation:
                issues.append("The task was a workspace mutation, but no real workspace mutation tool was executed.")

            content_mutation_tools = {"write_file", "append_file", "replace_text"}

            used_content_mutation = any(tool in content_mutation_tools for tool in executed_tools)

            if not created_paths and not deleted_paths and not used_content_mutation:
                issues.append("The task was a workspace mutation, but workspace_before and workspace_after show no " \
                                "workspace path change, and no content-changing tool was executed.")

        return issues
    
    def run(self, conversation_id: int, step_index: int, user_task: str, execution_response: dict, workspace_before: list = None, workspace_after: list = None,
            route: dict = None) -> Message:
        try:
            self.validate_input(user_task=user_task, execution_trace=execution_response, conversation_id=conversation_id, step_index=step_index, route=route)

            review_evidence = self.build_review_evidence(executor_response=execution_response, workspace_before=workspace_before,
                                                         workspace_after=workspace_after, route=route)
            
            deterministic_issues = self.deterministic_workspace_checks(user_task=user_task, review_evidence=review_evidence)

            if deterministic_issues:
                review_response = {
                    "accepted": False,
                    "review_summary": "Deterministic reviewer checks found workspace problems.",
                    "issues": deterministic_issues
                }
            else:
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

        return self.get_message(conversation_id=conversation_id, step_index=step_index, receiver="coordinator", target_agent=target_agent,
                                message_type="review_result", status=status, response=review_response, visibility="internal")
