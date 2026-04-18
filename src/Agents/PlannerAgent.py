from OllamaClient import OllamaClient
from BaseAgent import Agent
from message import Message

class PlannerAgent(Agent):
    #Class field,
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
                            "args": {"type": "object"}
                        },
                        "required": ["id", "tool", "args"]
                    }
                }
            },
            "required": ["goal", "steps"] #Helps llm know what keys should it have
        }
    
    def __init__(self, llm_client:OllamaClient):
        self.llm_client = llm_client
        super().__init__("planner")

    def build_messages(self,planner_input: dict) -> list[dict]:
        '''
            This method will build the prompt into a list of messages so that it is passed onto the LLM through the messages parameter of ollama's chat method
            which is a list of dictionaries
        '''
        return [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": f"""
                            Task:
                            {planner_input["task"]}
                            Relevant memory:
                            {planner_input["context"]}
                            Recent conversation:
                            {planner_input["recent_messages"]}
                            Available tools:
                            {planner_input["tools"]}
                    """
            }
        ]
    
    def validate_planner_input(self,planner_input:dict):
        required_keys = ["task","context","recent_messages","tools","conversation_id","step_index"]

        missing_keys = [key for key in required_keys if key not in planner_input]
        if missing_keys:
            raise ValueError(f"Missing planner_input keys: {missing_keys}")

    def validate_llm_response(self, response: dict):
        if not isinstance(response, dict):
            raise ValueError("Plan response must be a dictionary")

        required_top_level_fields = {"goal": str, "steps": list,}
        for field, expected_type in required_top_level_fields.items():
            if field not in response:
                raise ValueError(f"Plan response missing '{field}'")

            if not isinstance(response[field], expected_type):
                raise ValueError(f"Plan response field '{field}' must be {expected_type.__name__}")

        required_step_fields = {"id": int, "tool": str, "args": dict,}
        for step in response["steps"]:
            if not isinstance(step, dict):
                raise ValueError("Each step must be a dictionary")

            for field, expected_type in required_step_fields.items():
                if field not in step:
                    raise ValueError(f"Step missing '{field}'")

                if not isinstance(step[field], expected_type):
                    raise ValueError(f"Step field '{field}' must be {expected_type.__name__}")

    def get_plan(self, planner_input:dict) -> Message:
        ''' 
            planner_input would be a dictionary which will contain the task, context, recent_msgs and tools available so that all these info 
            Which are given by the coordinator will be assembled into one prompt for the LLM to reason about.
        '''

        #Try-catch block is done so that if the llm does not generate some response (or something internal breaks) then it fails
        try:
            self.validate_planner_input(planner_input)

            messages = self.build_messages(planner_input)
            
            response = self.llm_client.invoke_json(messages,stream=False,schema=self.SCHEMA)
            status = 'completed'
            target_agent = 'executor'
           
        except Exception as e:
            response = {"error": str(e)}
            status = 'failed'
            target_agent = None

        return self.get_message(conversation_id=planner_input["conversation_id"], step_index=planner_input["step_index"], sender = self.name,
                                receiver="coordinator", target_agent=target_agent, message_type="plan", status=status, response=response, visibility="internal"
                                ) 
