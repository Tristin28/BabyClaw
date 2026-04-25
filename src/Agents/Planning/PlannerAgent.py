from src.OllamaClient import OllamaClient
from src.Agents.BaseAgent import Agent
from src.message import Message
from src.Agents.Planning.PlanCompiler import PlanCompiler

class PlannerAgent(Agent):
    #Class field
    SCHEMA = {
            "type": "object", #Helps llm know what type the response will be converted to
            "properties": { #following is what actually is sent 
                "goal": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                            "depends_on": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            }
                        },
                        "required": ["id", "tool", "args","depends_on"]
                    }
                },
                "planning_rationale": {"type": "string"}
            },
            "required": ["goal", "steps", "planning_rationale"] #Helps llm know what keys should it have
        }
    
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
                You do NOT invent tools, files, arguments, or step dependencies.
                
                If the user gives an ambiguous file name without extension, do not guess the extension.
                Use list_dir first to find the closest matching file.

                Core planning rules:
                1. Use only tools from the Available tools list.
                2. Every step must map to one executable tool.
                3. Every step must fill args using that tool's args_schema exactly.
                4. Prefer the smallest valid plan.
                5. If the task can be solved with one tool call, create exactly one step.
                6. Do not add extra checking, reading, listing, or summarising steps unless they are required to complete the task.
                7. Step ids must start from 1 and increase in order: 1, 2, 3, ...
                8. Never use step id 0.
                9. Every step must include depends_on because the schema requires it.
                10. Always set depends_on to [].
                11. A deterministic compiler will infer the true dependencies from *_step arguments.
                12. Do not manually place dependencies inside depends_on.

                Argument rules:
                13. If the user directly gives an argument value, place that raw value directly inside args.
                14. If an argument must come from a previous tool result, use an argument name ending with _step.
                15. A *_step argument must contain the integer id of an earlier step.
                16. A *_step argument must never contain a filename, raw text, summary, or tool name.
                17. Only use *_step when the value really comes from a previous step.
                18. Do not invent a previous step just to use *_step.

                Dependency reasoning:
                19. Tool order is created through *_step arguments, not through depends_on.
                20. If step 2 needs the output of step 1, write this in args, for example: {"source_step": 1}.
                21. Still keep depends_on as [].
                22. The compiler will later transform depends_on from [] into [1].
                23. If no args contain *_step, then the step is independent.

                Failure rule:
                24. If no available tool can solve the task, return a valid JSON plan with:
                    - goal explaining that the task cannot be solved with available tools
                    - steps as an empty list
                    - planning_rationale briefly explaining why no tool can solve it

                Rationale rule:
                25. planning_rationale must be short.
                26. Do not include private chain-of-thought.
                27. Only explain briefly why the selected tools are needed.

                Output rule:
                28. Return only valid JSON matching the schema.
                29. Do not include markdown.
                30. Do not include explanations outside the JSON.

                Context usage rules:
                31. The CURRENT TASK is the main instruction.
                32. Recent conversation and semantic memory are only supporting context.
                33. Use context only when it helps resolve unclear references such as:
                - "it"
                - "that file"
                - "same as before"
                - "continue"
                - "previous task"
                34. Ignore context if the current task is already clear.
                35. Ignore context if it conflicts with the current task.
                36. Never create extra tool steps just because something appears in context.
                37. Never use context to invent file names, arguments, or dependencies.
                38. If the user gives the argument directly in the current task, use that value instead of context.

                Correct example 1:
                User task: append to hello.txt by saying hey

                

                {
                "goal": "Append text to hello.txt",
                "steps": [
                    {
                    "id": 1,
                    "tool": "append_file",
                    "args": {
                        "path": "hello.txt",
                        "text": "hey"
                    },
                    "depends_on": []
                    }
                ],
                "planning_rationale": "The user provided both the file path and text, so one append step is enough."
                }

                Correct example 2:
                User task: summarise hello.txt

                {
                "goal": "Read and summarise hello.txt",
                "steps": [
                    {
                    "id": 1,
                    "tool": "read_file",
                    "args": {
                        "path": "hello.txt"
                    },
                    "depends_on": []
                    },
                    {
                    "id": 2,
                    "tool": "summarise_txt",
                    "args": {
                        "source_step": 1
                    },
                    "depends_on": []
                    }
                ],
                "planning_rationale": "The file must first be read, then the produced text can be summarised."
                }

                Invalid examples:
                Do not write:
                {
                "source_step": "hello.txt"
                }

                Do not write:
                {
                "source_step": "file contents here"
                }

                Do not write:
                {
                "depends_on": [1]
                }

                Do not write:
                {
                "id": 0
                }

                Remember:
                - args decide data flow.
                - *_step arguments create dependencies.
                - depends_on must always be [] in your response.
                - the compiler will build the real depends_on list later.
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
        
   
    def run(self, planner_input:dict) -> Message:
        ''' 
            planner_input would be a dictionary which will contain the task, context, recent_msgs and tools available so that all these info 
            Which are given by the coordinator will be assembled into one prompt for the LLM to reason about.
        '''

        #Try-catch block is done so that if the llm does not generate some response (or something internal breaks) then it fails
        try:
            self.validate_planner_input(planner_input)

            messages = self.build_messages(planner_input)
            
            raw_response = self.llm_client.invoke_json(messages,stream=False,schema=self.SCHEMA)

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
    
    