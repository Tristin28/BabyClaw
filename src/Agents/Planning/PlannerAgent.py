from src.OllamaClient import OllamaClient
from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Agents.Planning.PlanCompiler import PlanCompiler

class PlannerAgent(Agent):
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
                "content": """
                    You are a planning agent.

                    Your job is to convert the user's task into a minimal structured execution plan using only the provided tools.

                    You do NOT execute tools.
                    You do NOT answer the user directly.
                    You do NOT invent tools, arguments, files, or dependencies.

                    Return only a valid JSON plan that matches the required schema.

                    --------------------------------------------------
                    Planning rules

                    1. Use only tools from the Available tools list.
                    2. Every step must correspond to exactly one tool.
                    3. Fill arguments exactly as required by each tool's args_schema.
                    4. Prefer the smallest valid plan that solves the task.
                    5. Do not add unnecessary steps.
                    6. If the user already provides argument values, use them directly.
                    7. If an argument depends on a previous step result, use a *_step reference.

                    Example:

                    If step 2 needs the output of step 1:

                    {
                    "id": 2,
                    "tool": "summarise_txt",
                    "args": {"text_step": 1}
                    }

                    --------------------------------------------------
                    Argument rules

                    8. Use *_step only when the argument comes from a previous step result.
                    9. *_step must always reference an earlier step id.
                    10. *_step must never contain filenames or raw text.
                    11. Do not invent earlier steps just to use *_step.

                    --------------------------------------------------
                    File handling rules

                    12. If the user provides an exact filename, use it directly.
                    13. If the filename is ambiguous, check Workspace contents before choosing.
                    14. Do not guess file extensions. If the user refers to a file without an exact filename or extension, use find_file first.
                        - If the user provides an exact filename, do not use find_file.
                    15. If the correct file cannot be identified safely, do not invent one.
                    
                    --------------------------------------------------
                    Context usage rules

                    16. Treat the CURRENT TASK as the primary instruction.
                    17. Use memory or conversation history only to resolve references like:
                        - "it"
                        - "that file"
                        - "same as before"
                        - "continue"
                    18. Ignore memory if it conflicts with the current task.
                    19. Never create extra steps just because something appears in memory.

                    --------------------------------------------------
                    Failure rule

                    20. If no available tool can solve the task:

                    Return:

                    {
                    "goal": "Explain why the task cannot be completed using available tools",
                    "steps": [],
                    "planning_rationale": "Brief explanation of why no available tool can solve the task"
                    }

                    --------------------------------------------------
                    Output format rules

                    21. Step ids must start at 1 and increase sequentially.
                    22. Do not include depends_on.
                    23. Do not include explanations outside JSON.
                    24. Return only valid JSON.
                    25. planning_rationale must be short.

                    --------------------------------------------------
                    Example 1

                    User task:
                    append to hello.txt by saying hey

                    Correct output:

                    {
                    "goal": "Append text to hello.txt",
                    "steps": [
                            {
                            "id": 1,
                            "tool": "append_file",
                            "args": {
                                "path": "hello.txt",
                                "content": "hey"
                                }
                            }
                        ],
                    "planning_rationale": "The user provided the file path and text, so one append step is enough."
                    }

                    --------------------------------------------------
                    Example 2

                    User task:
                    summarise hello.txt

                    Correct output:

                    {
                    "goal": "Read and summarise hello.txt",
                    "steps": [
                        {
                            "id": 1,
                            "tool": "read_file",
                            "args": {
                                "path": "hello.txt"
                            }
                        },
                        {
                            "id": 2,
                            "tool": "summarise_txt",
                            "args": {
                                "text_step": 1
                            }
                        }
                    ],
                    "planning_rationale": "The file must first be read, then summarised."
                    }
                """
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

            direct_properties = {}
            direct_required = []

            step_properties = {}
            step_required = []

            for arg_name, arg_spec in args_schema.items(): #arg_name= the function's argument name and arg_spec represents what tool_description says about it 
                arg_type = arg_spec["type"]
                is_chainable = arg_spec.get("step_chainable", False)

                direct_properties[arg_name] = {"type": arg_type}
                direct_required.append(arg_name)

                if is_chainable:
                    step_arg_name = f"{arg_name}_step"
                    step_properties[step_arg_name] = {"type": "integer"}
                    step_required.append(step_arg_name)

            arg_variants = []

            arg_variants.append({
                "type": "object",
                "properties": direct_properties,
                "required": direct_required,
                "additionalProperties": False
            })

            # Step-derived version, only if tool has chainable args
            if step_properties:
                arg_variants.append({
                    "type": "object",
                    "properties": step_properties,
                    "required": step_required,
                    "additionalProperties": False
                })

            tool_variants.append({
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tool": {
                        "type": "string",
                        "enum": [tool_name]
                    },
                    "args": {
                        "oneOf": arg_variants
                    }
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
