from src.llm.OllamaClient import OllamaClient
from src.agents.BaseAgent import Agent
from src.core.message import Message
from src.agents.planning.PlanCompiler import PlanCompiler
from src.agents.planning.PlannerPrompt import PLANNER_SYSTEM_PROMPT
from src.tools.utils import WorkspaceConfig
import json

class PlannerAgent(Agent):
    PLANNER_SYSTEM_PROMPT = PLANNER_SYSTEM_PROMPT
    def __init__(self, llm_client:OllamaClient, workspace_config: WorkspaceConfig):
        self.llm_client = llm_client
        super().__init__("planner") #setting the name field for when sending the message
        self.workspace_config = workspace_config

    def build_messages(self, planner_input: dict) -> list[dict]:
        messages = [{"role": "system", "content": PlannerAgent.PLANNER_SYSTEM_PROMPT}]

        sections = [f"Current task to plan:\n{planner_input['task']}"]

        route = planner_input.get("route", {})
        if route:
            sections.append(
                f"COORDINATOR ROUTE (authoritative scope - do not exceed):\n"
                f"task_type = {route.get('task_type')}\n"
                f"memory_mode = {route.get('memory_mode')}\n"
                f"use_recent_messages = {route.get('use_recent_messages')}\n"
                f"use_workspace = {route.get('use_workspace')}\n\n"
                f"The Coordinator already selected the only tools and context available. "
                f"You must stay inside this route. Use only the available tools shown."
            )

        if planner_input["k_recent_messages"]:
            recent_text = "\n".join(
                f"{msg['sender']}: {msg['content']}"
                for msg in planner_input["k_recent_messages"]
                if msg.get("sender") in {"user", "assistant"} and msg.get("content")
            )
            sections.append(f"Recent conversation (background only):\n{recent_text}")

        if planner_input["context"]:
            sections.append(f"Relevant memory:\n{planner_input['context']}")

        context_resolution = planner_input.get("context_resolution") or {}
        if context_resolution.get("has_references"):
            sections.append(self.build_context_resolution_section(context_resolution))

        requested_paths = planner_input.get("requested_paths") or []
        allowed_parent_dirs = planner_input.get("allowed_parent_dirs") or []

        if requested_paths:
            sections.append(
                "HARD USER CONSTRAINTS:\n"
                f"requested_paths = {requested_paths}\n"
                f"allowed_parent_dirs = {allowed_parent_dirs}\n\n"
                "Rules:\n"
                "- You must use these exact requested paths for file mutations.\n"
                "- You may only create parent directories listed in allowed_parent_dirs.\n"
                "- Do not infer, rename, simplify, or replace requested paths.\n"
                "- Do not use code.py, hello_world.py, main.py, output.py, or any other filename."
            )

        sections.append(f"Available tools:\n{planner_input['tools']}")

        if planner_input["workspace_contents"]:
            sections.append(f"Workspace content:\n{planner_input['workspace_contents']}")

        messages.append({"role": "user", "content": "\n\n".join(sections)})
        return messages
    
    def build_context_resolution_section(self, context_resolution: dict) -> str:
        '''
            Render ContextResolver output as a deterministic block the planner
            can rely on. The planner must use these resolutions instead of
            treating pronouns or contextual phrases as literal content.
        '''
        resolved = context_resolution.get("resolved_references", []) or []
        unresolved = context_resolution.get("unresolved_references", []) or []
        planner_context = context_resolution.get("planner_context", {}) or {}

        lines = ["RESOLVED CONTEXTUAL REFERENCES (authoritative; the Coordinator already mapped these):"]

        if resolved:
            for ref in resolved:
                summary = f"- '{ref.get('phrase')}' -> {ref.get('source_type')}"
                if ref.get("path"):
                    summary += f" (path: {ref['path']})"
                if ref.get("content") is not None:
                    preview = ref["content"]
                    if isinstance(preview, str) and len(preview) > 200:
                        preview = preview[:200] + "..."
                    summary += f" (content preview: {preview!r})"
                lines.append(summary)
        else:
            lines.append("- (none resolved)")

        if unresolved:
            lines.append("Unresolved references (do not invent values):")
            for ref in unresolved:
                lines.append(f"- '{ref.get('phrase')}' could not be resolved: {ref.get('reason')}")

        if context_resolution.get("should_ask_clarification"):
            lines.append(
                "Coordinator could not resolve at least one reference. Do not invent missing "
                "content or paths. If essential context is missing, plan a direct_response "
                "asking the user for clarification."
            )

        if planner_context:
            lines.append("Structured planner_context:")
            lines.append(json.dumps(planner_context, indent=2, default=str))

        lines.append(
            "Rules:\n"
            "- Use these resolved references instead of treating pronouns or contextual phrases as literal text.\n"
            "- Do not write the literal word 'it', 'this', 'that', 'them', or 'those' as file content.\n"
            "- If resolved_content is provided, that is the content the user wants saved.\n"
            "- If resolved_file_path is provided, that is the file the user is referring to.\n"
        )

        return "\n".join(lines)

    def validate_planner_input(self,planner_input:dict):
        required_keys = ["task","context","k_recent_messages","tools","conversation_id","step_index"]

        missing_keys = [key for key in required_keys if key not in planner_input]
        if missing_keys:
            raise ValueError(f"Missing planner_input keys: {missing_keys}")
        
        
    def build_schema(self, tools: list[dict]) -> dict:
        """
            Builds a dynamic JSON schema based on the available planner tools, this is done so that hallucinated tool names and invalid arguments are prevented
        """
        tool_names = [tool["name"] for tool in tools]

        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string"
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "minimum": 1
                            },
                            "tool": {
                                "type": "string",
                                "enum": tool_names
                            },
                            "args": {
                                "type": "object"
                            }
                        },
                        "required": ["id", "tool", "args"],
                        "additionalProperties": False
                    }
                },
                "planning_rationale": {
                    "type": "string"
                }
            },
            "required": ["goal", "steps", "planning_rationale"],
            "additionalProperties": False
        }

    def run(self, planner_input:dict) -> Message:
        ''' 
            planner_input would be a dictionary which will contain the task, context, recent_msgs and tools available so that all these info 
            Which are given by the coordinator will be assembled into one prompt for the LLM to reason about.
        '''

        #Try-catch block is done so that if the llm does not generate some response (or something internal breaks) then it fails
        try:
            self.validate_planner_input(planner_input)

            messages = self.build_messages(planner_input)

            schema = self.build_schema(planner_input["tools"])
            raw_response = self.llm_client.invoke_json(messages,stream=False,schema=schema)

            compiler = PlanCompiler(
                available_tools=planner_input["tools"],
                workspace_config=self.workspace_config,
                route=planner_input.get("route", {}),
                user_task=planner_input["task"],
                context_resolution=planner_input.get("context_resolution") or {}
            )
            response = compiler.compile(raw_response)

            status = 'completed'
            target_agent = 'executor'
           
        except Exception as e:
            response = {"error": str(e)}
            status = 'failed'
            target_agent = None

        return self.get_message(conversation_id=planner_input["conversation_id"], step_index=planner_input["step_index"],
                                receiver="coordinator", target_agent=target_agent, message_type="plan", status=status, response=response, visibility="internal"
                                ) 
