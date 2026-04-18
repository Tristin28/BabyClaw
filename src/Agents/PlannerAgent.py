from OllamaClient import OllamaClient
from BaseAgent import Agent
from message import Message

class PlannerAgent(Agent):
    #Class field
    SCHEMA = {
            "type": "object",
            "properties": {
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
            "required": ["goal", "steps"]
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
    
    def get_plan(self, planner_input:dict) -> Message:
        ''' 
            planner_input would be a dictionary which will contain the task, context, recent_msgs and tools available so that all these info 
            Which are given by the coordinator will be assembled into one prompt for the LLM to reason about.
        '''
        messages = self.build_messages(planner_input)

        try:
            #Try catch block is done so that if the llm does not generate some response (or something internal breaks) then it fails
            response = self.llm_client.invoke_json(messages,stream=False,schema=self.SCHEMA)
            status = 'completed'
            return self.get_message(conversation_id=planner_input["conversation_id"], step_index=planner_input["step_index"], sender = self.name,
                                receiver="coordinator", target_agent="executor", message_type="plan", status=status, response=response, visibility="internal"
                                ) 
        except Exception:
            response = {"error": "planner failed"}
            status = 'failed'
            return self.get_message(conversation_id=planner_input["conversation_id"], step_index=planner_input["step_index"], sender = self.name,
                                receiver="coordinator", target_agent="none", message_type="plan", status=status, response=response, visibility="internal"
                                ) 
