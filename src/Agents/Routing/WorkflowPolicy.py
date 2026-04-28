from dataclasses import dataclass

@dataclass(frozen=True) 
class WorkflowPolicy:
    task_type: str
    tool_group: str
    memory_mode: str
    use_recent_messages: bool
    use_workspace: bool
    allow_mutations: bool


class WorkflowPolicyRegistry:
    """
        Central place where task types become workflow contracts, hence why it should never be modified i.e. why it is created as immutable

        This keeps the LLM flexible while the infrastructure decides scope.
    """

    DIRECT_RESPONSE_TOOLS = {"direct_response"}

    READ_FILE_TOOLS = {
        "read_file",
        "find_file",
        "find_file_recursive",
        "list_dir",
        "list_tree",
        "search_text"
    }

    SUMMARISE_FILE_TOOLS = {
        "read_file",
        "find_file",
        "find_file_recursive",
        "summarise_txt"
    }

    MUTATION_FILE_TOOLS = {
        "generate_content",
        "create_file",
        "write_file",
        "append_file",
        "delete_file",
        "replace_text",
        "create_dir",
        "delete_dir",
        "move_path",
        "copy_path",
        "find_file",
        "find_file_recursive",
        "list_dir",
        "list_tree",
        "read_file",
        "summarise_txt",
        "search_text"
    }

    POLICIES = {
        "direct_response": WorkflowPolicy(
            task_type="direct_response",
            tool_group="direct_response_tools",
            memory_mode="pinned_only",
            use_recent_messages=False,
            use_workspace=False,
            allow_mutations=False
        ),
        "memory_question": WorkflowPolicy(
            task_type="memory_question",
            tool_group="direct_response_tools",
            memory_mode="full",
            use_recent_messages=False,
            use_workspace=False,
            allow_mutations=False
        ),
        "contextual_followup": WorkflowPolicy(
            task_type="contextual_followup",
            tool_group="direct_response_tools",
            memory_mode="full",
            use_recent_messages=True,
            use_workspace=False,
            allow_mutations=False
        ),
        "workspace_read": WorkflowPolicy(
            task_type="workspace_read",
            tool_group="read_file_tools",
            memory_mode="none",
            use_recent_messages=False,
            use_workspace=True,
            allow_mutations=False
        ),
        "workspace_summarise": WorkflowPolicy(
            task_type="workspace_summarise",
            tool_group="summarise_file_tools",
            memory_mode="none",
            use_recent_messages=False,
            use_workspace=True,
            allow_mutations=False
        ),
        "workspace_mutation": WorkflowPolicy(
            task_type="workspace_mutation",
            tool_group="mutation_file_tools",
            memory_mode="none",
            use_recent_messages=False,
            use_workspace=True,
            allow_mutations=True
        ),
    }

    TOOL_GROUPS = {
        "direct_response_tools": DIRECT_RESPONSE_TOOLS,
        "read_file_tools": READ_FILE_TOOLS,
        "summarise_file_tools": SUMMARISE_FILE_TOOLS,
        "mutation_file_tools": MUTATION_FILE_TOOLS,
    }

    SAFE_FALLBACK_TASK_TYPE = "direct_response"

    @classmethod
    def get_policy(cls, task_type: str) -> WorkflowPolicy:
        return cls.POLICIES.get(
            task_type,
            cls.POLICIES[cls.SAFE_FALLBACK_TASK_TYPE]
        )

    @classmethod
    def allowed_tools_for_policy(cls, policy: WorkflowPolicy) -> set[str]:
        return cls.TOOL_GROUPS.get(policy.tool_group, cls.DIRECT_RESPONSE_TOOLS)

    @classmethod
    def build_route(cls, router_response: dict) -> dict:
        """
            Converts router classification into an infrastructure-owned route.
        """

        if not isinstance(router_response, dict):
            router_response = {}

        task_type = router_response.get("task_type", cls.SAFE_FALLBACK_TASK_TYPE)
        confidence = router_response.get("confidence", 0.0)

        if task_type not in cls.POLICIES:
            task_type = cls.SAFE_FALLBACK_TASK_TYPE

        policy = cls.get_policy(task_type)
        allowed_tools = cls.allowed_tools_for_policy(policy)

        return {
            "task_type": policy.task_type,
            "tool_group": policy.tool_group,
            "memory_mode": policy.memory_mode,
            "use_recent_messages": policy.use_recent_messages,
            "use_workspace": policy.use_workspace,
            "allow_mutations": policy.allow_mutations,
            "allowed_tools": sorted(allowed_tools),
            "router_confidence": confidence,
            "routing_reason": router_response.get("routing_reason", "")
        }