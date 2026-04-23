from src.OllamaClient import OllamaClient
from src.Agents.BaseAgent import Agent
from src.message import Message

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
                #This is setting the agent's personality and instructions
                "role": "system",
                "content": """ 
                            You are a planning agent. Your job is to break down complex user requests into a structured, step-by-step execution plan using only the provided tools.

                            Rules:
                            1. Analyze the Task, Relevant memory, and Recent conversation to understand the user's goal.
                            2. Create a logical sequence of actions.
                            3. If information is missing, create a step using only a valid available tool. Do not describe vague intentions; every step must map to an executable action.
                            4. Each step must map to a specific tool from the Available tools list.
                            5. Each tool in Available tools includes its expected arguments. Fill args exactly using that tool's args_schema.
                            6. If an argument such as source_step refers to previous tool output, its value must be the integer id of an earlier step, not raw text and not a file name.
                            7. If a step uses the output of an earlier step, include that earlier step id in both args and depends_on.
                            8. Do not hallucinate tools that are not provided. Never invent a new tool.
                            9. If no available tool can solve the task, return a failure response explaining that no suitable tool exists.
                            10. Also return a field called planning_rationale that briefly explains why the plan was chosen. This must be a short justification summary and not a full chain-of-thought explanation.
                            11. Each step must include a field called depends_on. This field must be a list of earlier step IDs that must be completed before the current step can run.
                            12. Return only valid JSON matching the schema.

                            Example:
                            If step 1 reads a file and step 2 summarises that file's text, then step 2 must look like:
                            {
                            "id": 2,
                            "tool": "summarise_txt",
                            "args": {"source_step": 1},
                            "depends_on": [1]
                            }
                            Do not write:
                            {"source_step": "document.txt"}
                            Do not write:
                            {"source_step": "file contents here"}
                          """
            },

            {
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
                    """
            }
        ]

        #Seperating the recenent messages underneath their respective role
        history_messages = [{"role": ("user" if msg["sender"] == "user" else "assistant"), "content": msg["content"]}
                            for msg in planner_input["k_recent_messages"]]
        messages.extend(history_messages)

        return messages

    def validate_planner_input(self,planner_input:dict):
        required_keys = ["task","context","k_recent_messages","tools","conversation_id","step_index"]

        missing_keys = [key for key in required_keys if key not in planner_input]
        if missing_keys:
            raise ValueError(f"Missing planner_input keys: {missing_keys}")
        
    def validate_dependencies(self, steps: list[dict]):
        step_ids = set()
        for step in steps:
            if step["id"] in step_ids:
                raise ValueError(f"Duplicate step id {step['id']}")
            step_ids.add(step["id"])

        for step in steps:
            current_id = step["id"]
            depends_on = step.get("depends_on", [])

            if not isinstance(depends_on, list):
                raise ValueError(f"Step {current_id} 'depends_on' must be a list")

            for dep_id in depends_on:
                if not isinstance(dep_id, int):
                    raise ValueError(f"Step {current_id} has non-integer dependency")

                if dep_id not in step_ids:
                    raise ValueError(f"Step {current_id} depends on unknown step {dep_id}")

                if dep_id == current_id:
                    raise ValueError(f"Step {current_id} cannot depend on itself")

                if dep_id >= current_id:
                    raise ValueError(f"Step {current_id} can only depend on earlier step IDs")

    def validate_llm_response(self, response: dict, available_tools: list[dict]):
        tool_names = [tool["name"] for tool in available_tools]

        if not isinstance(response, dict):
            raise ValueError("Plan response must be a dictionary")

        required_top_level_fields = {"goal": str, "steps": list, "planning_rationale": str}
        for field, expected_type in required_top_level_fields.items():
            if field not in response:
                raise ValueError(f"Plan response missing '{field}'")

            if not isinstance(response[field], expected_type):
                raise ValueError(f"Plan response field '{field}' must be {expected_type.__name__}")

        required_step_fields = {"id": int, "tool": str, "args": dict, "depends_on": list}
        for step in response["steps"]:
            if not isinstance(step, dict):
                raise ValueError("Each step must be a dictionary")

            for field, expected_type in required_step_fields.items():
                if field not in step:
                    raise ValueError(f"Step missing '{field}'")

                if not isinstance(step[field], expected_type):
                    raise ValueError(f"Step field '{field}' must be {expected_type.__name__}")
            
            if step["tool"] not in tool_names:
                raise ValueError(f"Unknown tool '{step['tool']}' in plan")
            
        self.validate_dependencies(response["steps"])

    def run(self, planner_input:dict) -> Message:
        ''' 
            planner_input would be a dictionary which will contain the task, context, recent_msgs and tools available so that all these info 
            Which are given by the coordinator will be assembled into one prompt for the LLM to reason about.
        '''

        #Try-catch block is done so that if the llm does not generate some response (or something internal breaks) then it fails
        try:
            self.validate_planner_input(planner_input)

            messages = self.build_messages(planner_input)
            
            response = self.llm_client.invoke_json(messages,stream=False,schema=self.SCHEMA)
            self.validate_llm_response(response, planner_input["tools"])

            status = 'completed'
            target_agent = 'executor'
           
        except Exception as e:
            response = {"error": str(e)}
            status = 'failed'
            target_agent = None

        return self.get_message(conversation_id=planner_input["conversation_id"], step_index=planner_input["step_index"],
                                receiver="coordinator", target_agent=target_agent, message_type="plan", status=status, response=response, visibility="internal"
                                ) 
