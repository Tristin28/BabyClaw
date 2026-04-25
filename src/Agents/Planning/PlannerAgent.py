from src.OllamaClient import OllamaClient
from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Agents.Planning.PlanCompiler import PlanCompiler
from src.Agents.Planning.PlannerPrompt import PLANNER_SYSTEM_PROMPT

class PlannerAgent(Agent):
    PLANNER_SYSTEM_PROMPT = PLANNER_SYSTEM_PROMPT
    def __init__(self, llm_client:OllamaClient):
        self.llm_client = llm_client
        super().__init__("planner") #setting the name field for when sending the message

    def build_messages(self,planner_input: dict) -> list[dict]:
        '''
            This method will build the prompt into a list of messages so that it is passed onto the LLM through the messages parameter of ollama's chat method
            which is a list of dictionaries, where it is seperated by roles because it makes the llm understand better hence it improves planning reliability
        '''
        messages = [
            {
                #Agent's personality and instructions - system prompt
                "role": "system",
                "content": PlannerAgent.PLANNER_SYSTEM_PROMPT
            }
        ]

        #Seperating the recenent messages underneath their respective role
        history_messages = [{"role": ("user" if msg["sender"] == "user" else "assistant"), "content": msg["content"]} for msg in planner_input["k_recent_messages"]]
        messages.extend(history_messages)

        messages.append({
                #Describng what messages being sent by the user are and also any context 
                #(Relevanat memory) about it or what the user is asking  - this depends on memory retrieval and task
                #And also defining tools in this respective role because they are part of the current task environment
                "role": "user",
                "content": f"""
                            Task:
                            {planner_input["task"]}
                            Relevant memory: 
                            {planner_input["context"]}
                            Available tools:
                            {planner_input["tools"]}
                            Workspace content:
                            {planner_input["workspace_contents"]}
                    """
            })

        return messages

    def validate_planner_input(self,planner_input:dict):
        required_keys = ["task","context","k_recent_messages","tools","conversation_id","step_index"]

        missing_keys = [key for key in required_keys if key not in planner_input]
        if missing_keys:
            raise ValueError(f"Missing planner_input keys: {missing_keys}")
        
    def build_schema(self, tools: list[dict]) -> dict:
        """
        Builds a dynamic JSON schema based on the available planner tools, this is done so that hallucinated tool names and invalid arguments are prevented
        """
        tool_variants = []

        for tool in tools:
            tool_name = tool["name"]
            args_schema = tool["args_schema"]

            properties = {}
            required = []
            one_of_groups = []

            for arg_name, arg_spec in args_schema.items():
                arg_type = arg_spec["type"]
                is_chainable = arg_spec.get("step_chainable", False)

                if is_chainable:
                    step_arg_name = f"{arg_name}_step"

                    properties[arg_name] = {"type": arg_type}
                    properties[step_arg_name] = {"type": "integer"}

                    one_of_groups.append({
                        "oneOf": [
                            {"required": [arg_name]},
                            {"required": [step_arg_name]}
                        ]
                    })
                else:
                    properties[arg_name] = {"type": arg_type}
                    required.append(arg_name)

            args_object_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

            if one_of_groups:
                args_object_schema["allOf"] = one_of_groups

            tool_variants.append({
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tool": {
                        "type": "string",
                        "enum": [tool_name]
                    },
                    "args": args_object_schema
                },
                "required": ["id", "tool", "args"],
                "additionalProperties": False
            })

        return {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "oneOf": tool_variants
                    }
                },
                "planning_rationale": {"type": "string"}
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

            compiler = PlanCompiler(available_tools=planner_input["tools"])
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
