class PlanCompiler:
    """
    Main idea is that planner chooses the tools and args then this component will rewrites step ids by infering depends_on from args ending in _step.
    """

    REQUIRED_TOP_LEVEL_FIELDS = {
        "goal": str,
        "steps": list,
        "planning_rationale": str
    }

    REQUIRED_STEP_FIELDS = {
        "tool": str,
        "args": dict
    }

    def __init__(self, available_tools: list[dict]):
        self.available_tools = available_tools
        self.tool_names = {tool["name"] for tool in available_tools}
        self.tool_args_schema = {
            tool["name"]: tool["args_schema"]
            for tool in available_tools
        }
    
    def normalise_step_ids(self, raw_steps: list[dict]) -> list[dict]:
        """
            Rewrites planner step ids to deterministic order: 1..N and builds mapping old_id -> new_id
        """
        self.old_to_new_id = {}

        normalised_steps = []

        for new_id, step in enumerate(raw_steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"Step {new_id} must be a dictionary.")

            old_id = step.get("id")

            if isinstance(old_id, int):
                self.old_to_new_id[old_id] = new_id

            normalised_steps.append({
                "id": new_id,
                "tool": step.get("tool"),
                "args": step.get("args", {}),
                "depends_on": []  # rebuilt later
            })

        return normalised_steps
    
    def compile(self, raw_plan: dict) -> dict:
        """
        Main public method.
        This is the only method PlannerAgent needs to call.
        """
        self.validate_top_level_schema(raw_plan)

        normalised_steps = self.normalise_step_ids(raw_plan["steps"])
        normalised_steps = self.remap_step_arguments(normalised_steps)
        self.validate_step_args_are_allowed(normalised_steps)

        normalised_steps = self.infer_dependencies_from_step_args(normalised_steps)
        self.validate_dependencies(normalised_steps)

        return {
            "goal": raw_plan["goal"],
            "steps": normalised_steps,
            "planning_rationale": raw_plan["planning_rationale"]
        }

    def validate_top_level_schema(self, raw_plan: dict) -> None:
        if not isinstance(raw_plan, dict):
            raise ValueError("Planner response must be a dictionary.")

        for field, expected_type in self.REQUIRED_TOP_LEVEL_FIELDS.items():
            if field not in raw_plan:
                raise ValueError(f"Planner response missing top-level field: {field}")

            if not isinstance(raw_plan[field], expected_type):
                raise ValueError(
                    f"Planner response field '{field}' must be {expected_type.__name__}."
                )

        if len(raw_plan["steps"]) == 0:
            raise ValueError("Planner produced no executable steps.")


    def remap_step_arguments(self, steps: list[dict]) -> list[dict]:
        """
        Remaps args such as source_step from old planner ids to new compiler ids.
        """
        for step in steps:
            current_id = step["id"]
            args = step.get("args", {})

            if not isinstance(args, dict):
                raise ValueError(f"Step {current_id} args must be a dictionary.")

            remapped_args = {}

            for arg_name, arg_value in args.items():
                if arg_name.endswith("_step"):
                    remapped_args[arg_name] = self.remap_step_reference(
                        current_id=current_id,
                        arg_name=arg_name,
                        arg_value=arg_value
                    )
                else:
                    remapped_args[arg_name] = arg_value

            step["args"] = remapped_args

        return steps

    def remap_step_reference(self, current_id: int, arg_name: str, arg_value: int) -> int:
        """
        Validates and remaps one *_step argument.
        """
        if not isinstance(arg_value, int):
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' must reference an integer step id."
            )

        if arg_value <= 0:
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' cannot reference step {arg_value}."
            )

        if arg_value not in self.old_to_new_id:
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' references unknown step {arg_value}."
            )

        new_reference = self.old_to_new_id[arg_value]

        if new_reference >= current_id:
            raise ValueError(
                f"Step {current_id} argument '{arg_name}' must reference an earlier step."
            )

        return new_reference

    def infer_dependencies_from_step_args(self, steps: list[dict]) -> list[dict]:
        """
        We do not trust the planner's depends_on.
        We infer dependencies from arguments ending in _step.
        """
        for step in steps:
            dependencies = set()

            for arg_name, arg_value in step["args"].items():
                if arg_name.endswith("_step"):
                    dependencies.add(arg_value)

            step["depends_on"] = sorted(dependencies)

        return steps

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

    def validate_step_args_are_allowed(self, steps: list[dict]) -> None:
        for step in steps:
            step_id = step["id"]
            tool_name = step["tool"]
            args = step["args"]

            if tool_name not in self.tool_args_schema:
                raise ValueError(f"Step {step_id} uses unknown tool '{tool_name}'")
        
            args_schema = self.tool_args_schema[tool_name]

            for arg_name in args.keys():

                # Direct arg
                if not arg_name.endswith("_step"):
                    if arg_name not in args_schema:
                        raise ValueError(
                            f"Step {step_id} tool '{tool_name}' has unknown arg '{arg_name}'"
                        )
                   
                    expected_type = args_schema[arg_name]["type"]

                    if not self.matches_schema_type(args[arg_name], expected_type):
                        raise ValueError(
                            f"Step {step_id} tool '{tool_name}' arg '{arg_name}' must be {expected_type}"
                        )

                    continue

                # Step-derived arg
                base_arg = arg_name.removesuffix("_step")

                if base_arg not in args_schema:
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' has unknown step arg '{arg_name}'"
                    )

                if not args_schema[base_arg].get("step_chainable", False):
                    raise ValueError(
                        f"Step {step_id} tool '{tool_name}' arg '{base_arg}' cannot use _step"
                    )

    def validate_dependencies(self, steps: list[dict]) -> None:
        valid_step_ids = {step["id"] for step in steps}

        for step in steps:
            current_id = step["id"]

            for dep_id in step["depends_on"]:
                if not isinstance(dep_id, int):
                    raise ValueError(f"Step {current_id} has a non-integer dependency.")

                if dep_id not in valid_step_ids:
                    raise ValueError(f"Step {current_id} depends on unknown step {dep_id}.")

                if dep_id == current_id:
                    raise ValueError(f"Step {current_id} cannot depend on itself.")

                if dep_id > current_id:
                    raise ValueError(f"Step {current_id} cannot depend on a future step.")