from pathlib import Path
from pathlib import PurePosixPath
import re
from src.action_constants import MUTATION_TOOLS, MUTATION_PATH_ARGS, GENERATED_ARTIFACT_TERMS, CREATIVE_ARTIFACT_PREFIXES, CONTEXTUAL_REFERENCE_PRONOUNS, CONTEXTUAL_REFERENCE_PHRASES
from src.tools.utils import WorkspaceConfig

class PlanCompiler:
    REQUIRED_TOP_LEVEL_FIELDS = {
        "goal": str,
        "steps": list,
        "planning_rationale": str,
    }

    REQUIRED_STEP_FIELDS = {
        "tool": str,
        "args": dict,
    }

    MUTATION_TOOLS = MUTATION_TOOLS
    MUTATION_PATH_ARGS = MUTATION_PATH_ARGS

    def __init__(self, available_tools: list[dict], workspace_config: WorkspaceConfig = None, route: dict = None, 
                 user_task: str = "", context_resolution: dict = None):
        """
        available_tools should be the same planner-facing tool descriptions
        given to the PlannerAgent.

        workspace_config is used to validate path-like planner arguments before
        execution or permission. This means hallucinated paths like
        '../../etc/passwd' fail during plan compilation.
        """
        self.available_tools = available_tools
        self.workspace_config = workspace_config
        self.route = route
        self.user_task = user_task or ""
        self.context_resolution = context_resolution or {}
        self.tool_names = {
            tool["name"]
            for tool in available_tools
        }

        self.tool_args_schema = {
            tool["name"]: tool["args_schema"]
            for tool in available_tools
        }

        # Created during normalise_step_ids().
        # Maps planner-provided ids to compiler-normalised ids.
        self.old_to_new_id: dict[int, int] = {}

    def validate_workspace_paths(self, steps: list[dict]) -> None:
        """
        Validates direct path-like arguments before execution.

        This prevents planner-hallucinated paths from reaching:
        - permission requests,
        - tool execution,
        - rollback logic.

        Only direct string args are checked here.
        *_step args are skipped because their values come from previous tool
        outputs and are still checked by the runtime workspace sandbox.
        """
        if self.workspace_config is None:
            return

        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step.get("args", {})

            for arg_name, arg_value in args.items():
                if arg_name.endswith("_step"):
                    continue

                if arg_name not in self.MUTATION_PATH_ARGS:
                    continue

                if not isinstance(arg_value, str):
                    continue

                cleaned_value = arg_value.strip()

                if cleaned_value == "":
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{arg_name}' "
                        f"cannot be an empty path."
                    )

                is_windows_absolute = (
                    len(cleaned_value) >= 3
                    and cleaned_value[1] == ":"
                    and cleaned_value[2] in {"\\", "/"}
                )

                if Path(cleaned_value).is_absolute() or is_windows_absolute:
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{arg_name}' "
                        f"must be a relative workspace path, got '{arg_value}'."
                    )

                if tool_name in self.MUTATION_TOOLS and cleaned_value in {".", "./"}:
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{arg_name}' "
                        f"cannot target the workspace root directly."
                    )

                try:
                    self.workspace_config.resolve_workspace_path(cleaned_value)
                except Exception as e:
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{arg_name}' "
                        f"resolves outside the workspace or is unsafe: '{arg_value}'. "
                        f"Reason: {e}"
                    )

    def normalise_relative_path_text(self, path_text: str) -> str:
        cleaned = path_text.strip().strip("\"'`.,;:!?)]}")
        cleaned = cleaned.replace("\\", "/")

        while cleaned.startswith("./"):
            cleaned = cleaned[2:]

        parts = []
        for part in cleaned.split("/"):
            if part in {"", "."}:
                continue
            parts.append(part)

        return "/".join(parts)

    def extract_explicit_file_paths(self, user_task: str) -> list[str]:
        """
            Extract explicit file paths from the user task.

            This intentionally looks for file-like paths with an extension, including
            quoted names containing spaces. It does not infer generic directories.
        """
        if not isinstance(user_task, str) or user_task.strip() == "":
            return []

        explicit_paths = []
        seen = set()

        quoted_pattern = re.compile(r'["\']([^"\']*\.[A-Za-z0-9]{1,12})["\']')
        unquoted_pattern = re.compile(
            r"(?<![\w./-])"
            r"([A-Za-z0-9_./-]*[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,12})"
            r"(?![\w./-])"
        )

        for pattern in (quoted_pattern, unquoted_pattern):
            for match in pattern.finditer(user_task):
                candidate = self.normalise_relative_path_text(match.group(1))

                if not candidate:
                    continue

                if candidate.startswith("/") or candidate.startswith("../") or "/../" in candidate:
                    continue

                if candidate not in seen:
                    explicit_paths.append(candidate)
                    seen.add(candidate)

        return explicit_paths

    def allowed_paths_for_requested_files(self, requested_paths: list[str]) -> set[str]:
        allowed_paths = set()

        for requested_path in requested_paths:
            allowed_paths.add(requested_path)
            parent = PurePosixPath(requested_path).parent

            if str(parent) == ".":
                continue

            current_parts = []
            for part in parent.parts:
                current_parts.append(part)
                allowed_paths.add("/".join(current_parts))

        return allowed_paths

    def collect_planned_mutation_paths(self, steps: list[dict]) -> list[str]:
        planned_paths = []

        for step in steps:
            if step.get("tool") not in self.MUTATION_TOOLS:
                continue

            for arg_name, arg_value in step.get("args", {}).items():
                if arg_name.endswith("_step"):
                    continue

                if arg_name not in self.MUTATION_PATH_ARGS:
                    continue

                if not isinstance(arg_value, str):
                    continue

                planned_path = self.normalise_relative_path_text(arg_value)
                if planned_path:
                    planned_paths.append(planned_path)

        return planned_paths

    def validate_explicit_mutation_paths(self, steps: list[dict]) -> None:
        requested_paths = self.extract_explicit_file_paths(self.user_task)

        if not requested_paths:
            return

        allowed_paths = self.allowed_paths_for_requested_files(requested_paths)
        requested_path_text = requested_paths[0] if len(requested_paths) == 1 else ", ".join(requested_paths)

        for planned_path in self.collect_planned_mutation_paths(steps):
            if planned_path in allowed_paths:
                continue

            raise ValueError(
                f"Plan rejected: planned mutation path '{planned_path}' "
                f"does not match requested path '{requested_path_text}'."
            )
                
    def apply_safe_defaults(self, steps: list[dict]) -> list[dict]:
        '''
            Deterministic fallback for tools where a missing argument is almost always
            "the user gave no value". Lets the workflow proceed instead of failing
            because the planner forgot to emit content="".
        '''
        EMPTY_CONTENT_TOOLS = {"create_file", "write_file", "append_file"}

        for step in steps:
            tool_name = step["tool"]
            args = step.get("args", {})

            if tool_name in EMPTY_CONTENT_TOOLS:
                has_direct = "content" in args
                has_step = "content_step" in args

                if not has_direct and not has_step:
                    args["content"] = ""

            step["args"] = args

        return steps
    
    def repair_placeholder_chains(self, steps: list[dict]):
        '''
            When the planner emits placeholders like "{{summary}}", "[Content of X]"
            or "<placeholder>" in a chainable arg, do not fail. Instead, find the
            most recent earlier step that produces a string and rewrite the arg as
            "<arg>_step": <that_step_id>. Only fall back to raising when no candidate
            exists or the arg is not step-chainable.
        '''
        import re
        placeholder_patterns = [
            re.compile(r"\{\{.*?\}\}"),
            re.compile(r"\[[A-Z][^\]]*\]"),
            re.compile(r"<[A-Za-z][^>]*>"),
        ]

        #Tools whose return value is a string usable as content/text/prompt input.
        string_producing_tools = {
            "read_file",
            "summarise_txt",
            "direct_response",
            "generate_content",
            "find_file",
            "find_file_recursive",
            "replace_text",
        }

        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step.get("args", {})
            args_schema = self.tool_args_schema.get(tool_name, {})

            for arg_name in list(args.keys()):
                arg_value = args[arg_name]

                if not isinstance(arg_value, str):
                    continue

                if not any(pattern.search(arg_value) for pattern in placeholder_patterns):
                    continue

                arg_def = args_schema.get(arg_name)

                if not arg_def or not arg_def.get("step_chainable"):
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{arg_name}' contains a placeholder "
                        f"and is not step-chainable. Provide a real value."
                    )

                #Pick the most recent earlier step that produces a string.
                candidate_id = None
                for earlier in steps:
                    if earlier["id"] >= step_id:
                        break
                    if earlier["tool"] in string_producing_tools:
                        candidate_id = earlier["id"]

                if candidate_id is None:
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{arg_name}' contains a placeholder "
                        f"but no earlier step produces a string to chain from. "
                        f"Add a generate_content/read_file/summarise_txt step before this one."
                    )

                #Rewrite: drop the placeholder direct arg, insert a step reference.
                del args[arg_name]
                args[f"{arg_name}_step"] = candidate_id

            step["args"] = args

    def repair_bare_step_marker_args(self, steps: list[dict]) -> None:
        """
            Repair the narrow safe case where the planner writes:
            {"content": "content_step"}
            instead of:
            {"content_step": <previous_generate_content_step_id>}
        """
        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step.get("args", {})
            args_schema = self.tool_args_schema.get(tool_name, {})

            for arg_name, arg_value in list(args.items()):
                if not isinstance(arg_value, str):
                    continue

                bare_step_arg = f"{arg_name}_step"

                if arg_value.strip() != bare_step_arg:
                    continue

                arg_def = args_schema.get(arg_name)
                if not arg_def or not arg_def.get("step_chainable"):
                    continue

                if arg_name != "content":
                    continue

                candidate_ids = [
                    earlier["id"]
                    for earlier in steps
                    if earlier["id"] < step_id and earlier["tool"] == "generate_content"
                ]

                if len(candidate_ids) != 1:
                    continue

                del args[arg_name]
                args[bare_step_arg] = candidate_ids[0]

            step["args"] = args

    def extract_exact_written_text(self, user_task: str) -> str | None:
        if not isinstance(user_task, str):
            return None

        patterns = [
            r"\b(?:inside|into|in)\s+(?:it|this|that|a\s+text\s+file|the\s+text\s+file|a\s+file|the\s+file)\s+write\s+(.+)$",
            r"\bwrite\s+(.+?)\s+(?:inside|into|to|in)\s+(?:it|this|that|a\s+text\s+file|the\s+text\s+file|a\s+file|the\s+file|[\w./ -]+\.[A-Za-z0-9]{1,12})\b",
            r"\bcreate\s+(?:a\s+)?(?:text\s+)?file\s+(?:with|containing)\s+(.+)$",
            r"\bcreate\s+[\w./ -]+\.[A-Za-z0-9]{1,12}\s+with\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_task, flags=re.IGNORECASE)
            if not match:
                continue

            text = match.group(1).strip(" .,!?:;\"'")
            if not text:
                continue

            lowered_text = text.lower()

            if " about " in f" {lowered_text} ":
                continue

            if lowered_text.startswith(CREATIVE_ARTIFACT_PREFIXES):
                continue

            if any(term in lowered_text for term in GENERATED_ARTIFACT_TERMS):
                continue

            if lowered_text in CONTEXTUAL_REFERENCE_PRONOUNS:
                continue

            if lowered_text in CONTEXTUAL_REFERENCE_PHRASES:
                continue


            return text

        return None

    def enforce_exact_written_text(self, steps: list[dict]) -> None:
        """
            Literal user-provided file content is not planner-owned. The planner
            may infer a filename when none was provided, but it must not replace
            "write <NAME>" with generic content like "Hello, world!".
        """
        exact_text = self.extract_exact_written_text(self.user_task)
        if exact_text is None:
            return

        content_tools = {"create_file", "write_file", "append_file"}
        updated = False

        for step in steps:
            if step.get("tool") not in content_tools:
                continue

            args = step.get("args", {})
            if "content_step" in args:
                continue

            if "content" not in args:
                continue

            args["content"] = exact_text
            step["args"] = args
            updated = True

        if not updated:
            raise ValueError(
                f"Plan rejected: the user explicitly asked to write '{exact_text}', "
                "but no direct file-writing content argument used that text."
            )
        
    def enforce_resolved_content(self, steps: list[dict]) -> None:
        '''
            When ContextResolver mapped a pronoun like "it" to a concrete previous
            assistant response or generated content, the planner LLM is not
            trusted to copy that text into args["content"]. We deterministically
            rewrite content-writing steps so the resolved content actually lands
            in the file. Steps that chain from a generate_content step via
            content_step are left alone because their content comes from runtime.
        '''
        planner_context = self.context_resolution.get("planner_context") or {}
        resolved_content = planner_context.get("resolved_content")

        if not isinstance(resolved_content, str) or resolved_content == "":
            return

        content_tools = {"create_file", "write_file", "append_file"}
        placeholder_values = {"", "it", "this", "that", "them", "those"}

        for step in steps:
            if step.get("tool") not in content_tools:
                continue

            args = step.get("args", {})

            # A real chained value from a previous step wins; do not override it.
            if "content_step" in args:
                continue

            existing = args.get("content")
            if isinstance(existing, str) and existing.strip().lower() not in placeholder_values:
                # Literal user-provided text already enforced by
                # enforce_exact_written_text. Keep it.
                continue

            args["content"] = resolved_content
            step["args"] = args
                    
    def compile(self, raw_plan: dict) -> dict:
        self.validate_top_level_schema(raw_plan)
        self.validate_workspace_mutation_has_real_mutation(raw_plan=raw_plan, route=self.route)

        normalised_steps = self.normalise_step_ids(raw_plan["steps"])
        normalised_steps = self.remap_step_arguments(normalised_steps)

        normalised_steps = self.apply_safe_defaults(normalised_steps)
        self.enforce_exact_written_text(normalised_steps)
        self.enforce_resolved_content(normalised_steps)
        self.repair_placeholder_chains(normalised_steps)
        self.repair_bare_step_marker_args(normalised_steps)
        self.reject_fake_step_strings(normalised_steps)

        self.validate_step_args_are_allowed(normalised_steps)
        self.validate_workspace_paths(normalised_steps)
        self.validate_explicit_mutation_paths(normalised_steps)
        self.validate_required_args_present(normalised_steps)

        normalised_steps = self.infer_dependencies_from_step_args(normalised_steps)
        self.validate_dependencies(normalised_steps)

        return {
            "goal": raw_plan["goal"],
            "steps": normalised_steps,
            "planning_rationale": raw_plan["planning_rationale"],
        }

    def validate_top_level_schema(self, raw_plan: dict) -> None:
        if not isinstance(raw_plan, dict):
            raise ValueError("Planner response must be a dictionary.")

        for field, expected_type in self.REQUIRED_TOP_LEVEL_FIELDS.items():
            if field not in raw_plan:
                raise ValueError(
                    f"Planner response missing top-level field '{field}'."
                )

            if not isinstance(raw_plan[field], expected_type):
                raise ValueError(
                    f"Planner response field '{field}' must be "
                    f"{expected_type.__name__}, got {type(raw_plan[field]).__name__}."
                )

        if len(raw_plan["steps"]) == 0:
            raise ValueError("Planner produced no executable steps.")

    def normalise_step_ids(self, raw_steps: list[dict]) -> list[dict]:
        self.old_to_new_id = {}
        normalised_steps = []

        for new_id, step in enumerate(raw_steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(
                    f"Step {new_id} must be a dictionary."
                )

            self.validate_raw_step_shape(step, new_id)

            old_id = step.get("id")

            if isinstance(old_id, int):
                if old_id in self.old_to_new_id:
                    raise ValueError(
                        f"Planner produced duplicate step id {old_id}."
                    )

                self.old_to_new_id[old_id] = new_id
            else:
                # Defensive fallback in case a planner response reaches the compiler without an id.
                self.old_to_new_id[new_id] = new_id

            normalised_steps.append({
                "id": new_id,
                "tool": step["tool"],
                "args": step.get("args", {}),
                # We never trust planner-provided depends_on.
                # Dependencies are rebuilt later from *_step args.
                "depends_on": [],
            })

        return normalised_steps

    def validate_raw_step_shape(self, step: dict, position: int) -> None:
        for field, expected_type in self.REQUIRED_STEP_FIELDS.items():
            if field not in step:
                raise ValueError(
                    f"Step {position} is missing required field '{field}'."
                )

            if not isinstance(step[field], expected_type):
                raise ValueError(
                    f"Step {position} field '{field}' must be "
                    f"{expected_type.__name__}, got {type(step[field]).__name__}."
                )

    def remap_step_arguments(self, steps: list[dict]) -> list[dict]:
        for step in steps:
            current_id = step["id"]
            args = step.get("args", {})

            if not isinstance(args, dict):
                raise ValueError(
                    f"Step {current_id} args must be a dictionary."
                )

            remapped_args = {}

            for arg_name, arg_value in args.items():
                if arg_name.endswith("_step"):
                    remapped_args[arg_name] = self.remap_step_reference(
                        current_id=current_id,
                        arg_name=arg_name,
                        arg_value=arg_value,
                    )
                else:
                    remapped_args[arg_name] = arg_value

            step["args"] = remapped_args

        return steps

    def remap_step_reference(self, current_id: int, arg_name: str, arg_value: int,) -> int:
        if not isinstance(arg_value, int) or isinstance(arg_value, bool):
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' must reference "
                f"an integer step id, got {type(arg_value).__name__}."
            )

        if arg_value <= 0:
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' cannot reference "
                f"invalid step id {arg_value}."
            )

        if arg_value not in self.old_to_new_id:
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' references "
                f"unknown step {arg_value}."
            )

        new_reference = self.old_to_new_id[arg_value]

        if new_reference >= current_id:
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' must reference "
                f"an earlier step, but references step {new_reference}."
            )

        return new_reference

    def reject_fake_step_strings(self, steps: list[dict]) -> None:
        fake_step_markers = [ "_step:", "_step=", "_step ->","_step=>"]

        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step.get("args", {})

            for arg_name, arg_value in args.items():
                if not isinstance(arg_value, str):
                    continue

                compact_value = arg_value.replace(" ", "").lower()

                contains_fake_marker = any(
                    marker.replace(" ", "") in compact_value
                    for marker in fake_step_markers
                )

                looks_like_step_reference = (
                    compact_value.endswith("_step")
                    or compact_value.startswith("step:")
                    or compact_value.startswith("step_")
                )

                if contains_fake_marker or looks_like_step_reference:
                    expected_step_arg = f"{arg_name}_step"

                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' has a fake step "
                        f"reference inside normal argument '{arg_name}'. "
                        f"Wrong: {{'{arg_name}': '{arg_value}'}}. "
                        f"If this value comes from a previous step, use "
                        f"{{'{expected_step_arg}': <previous_step_id>}} instead."
                    )

    def validate_step_args_are_allowed(self, steps: list[dict]) -> None:
        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step["args"]

            if tool_name not in self.tool_args_schema:
                raise ValueError(f"Step {step_id} uses unknown tool '{tool_name}'.")

            if not isinstance(args, dict):
                raise ValueError(f"Step {step_id} args must be a dictionary.")
            
            args_schema = self.tool_args_schema[tool_name]

            for arg_name, arg_value in args.items():
                if arg_name.endswith("_step"):
                    self.validate_step_arg(step_id=step_id, tool_name=tool_name, arg_name=arg_name, arg_value=arg_value, args_schema=args_schema)
                else:
                    self.validate_direct_arg(step_id=step_id, tool_name=tool_name, arg_name=arg_name, arg_value=arg_value, args_schema=args_schema)

    def validate_required_args_present(self, steps: list[dict]):
        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step["args"]

            args_schema = self.tool_args_schema[tool_name]

            for arg_name, arg_info in args_schema.items():
                required = arg_info.get("required", True)

                if not required:
                    continue

                direct_arg_exists = arg_name in args
                step_arg_exists = f"{arg_name}_step" in args

                if not direct_arg_exists and not step_arg_exists:
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' is missing required argument "
                        f"'{arg_name}'. Expected either '{arg_name}' or '{arg_name}_step'."
                    )
                
    def validate_workspace_mutation_has_real_mutation(self, raw_plan: dict, route: dict = None) -> None:
        if not route:
            return

        if route.get("task_type") != "workspace_mutation":
            return

        real_mutation_tools = {
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

        steps = raw_plan.get("steps", [])

        has_real_mutation = any(
            step.get("tool") in real_mutation_tools
            for step in steps
        )

        if not has_real_mutation:
            raise ValueError(
                "Workspace mutation task must include at least one real workspace mutation tool. "
                "generate_content alone does not complete a workspace task."
            )
                
    def validate_direct_arg(self,step_id: int, tool_name: str, arg_name: str, arg_value, args_schema: dict):
        if arg_name not in args_schema:
            raise ValueError(
                f"Step {step_id} tool '{tool_name}' has unknown arg '{arg_name}'."
            )

        expected_type = args_schema[arg_name]["type"]

        if not self.matches_schema_type(arg_value, expected_type):
            raise ValueError(
                f"Step {step_id} tool '{tool_name}' arg '{arg_name}' must be "
                f"{expected_type}, got {type(arg_value).__name__}. "
                f"If this value should come from a previous step, use "
                f"'{arg_name}_step': <step_id> instead."
            )

    def validate_step_arg(self, step_id: int, tool_name: str, arg_name: str, arg_value, args_schema: dict):
        base_arg = arg_name.removesuffix("_step")

        if base_arg not in args_schema:
            raise ValueError(
                f"Step {step_id} tool '{tool_name}' has unknown step arg "
                f"'{arg_name}'. Base arg '{base_arg}' does not exist."
            )

        if not args_schema[base_arg].get("step_chainable", False):
            raise ValueError(
                f"Step {step_id} tool '{tool_name}' arg '{base_arg}' cannot "
                f"use _step because step_chainable=False."
            )

        if not isinstance(arg_value, int) or isinstance(arg_value, bool):
            raise ValueError(
                f"Step {step_id} tool '{tool_name}' arg '{arg_name}' must be "
                f"an integer step id, got {type(arg_value).__name__}."
            )

    def matches_schema_type(self, value, expected_type: str) -> bool:
        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)

        if expected_type == "array":
            return isinstance(value, list)

        if expected_type == "object":
            return isinstance(value, dict)

        if expected_type == "boolean":
            return isinstance(value, bool)
        
        return True

    def infer_dependencies_from_step_args(self, steps: list[dict]) -> list[dict]:
        for step in steps:
            dependencies = set()

            for arg_name, arg_value in step["args"].items():
                if arg_name.endswith("_step"):
                    dependencies.add(arg_value)

            step["depends_on"] = sorted(dependencies)

        return steps

    def validate_dependencies(self, steps: list[dict]) -> None:
        valid_step_ids = {
            step["id"]
            for step in steps
        }

        for step in steps:
            current_id = step["id"]

            for dep_id in step["depends_on"]:
                if not isinstance(dep_id, int) or isinstance(dep_id, bool):
                    raise ValueError(f"Step {current_id} has a non-integer dependency.")

                if dep_id not in valid_step_ids:
                    raise ValueError(
                        f"Step {current_id} depends on unknown step {dep_id}."
                    )

                if dep_id == current_id:
                    raise ValueError(f"Step {current_id} cannot depend on itself.")

                if dep_id > current_id:
                    raise ValueError(f"Step {current_id} cannot depend on future step {dep_id}.")
