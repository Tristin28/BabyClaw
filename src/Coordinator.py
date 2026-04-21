from src.Agents.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.ReviewerAgent import ReviewerAgent
from src.message import Message

'''
    This is the agent which manages workflow, and configures the respective agents i.e. it acts like the runner method
'''
class Coordinator():
    def __init__(self,planner:PlannerAgent, executor:ExecutorAgent, reviewer:ReviewerAgent, memory:MemoryAgent, 
                 planner_tool_descriptions: list[dict], tool_registry: dict):
        self.planner = planner
        self.planner_tool_descriptions = planner_tool_descriptions
        self.executor = executor
        self.tool_registry = tool_registry
        self.reviewer = reviewer
        self.memory = memory

    def handle_user_request(self, conversation_id: int, user_task: str) -> Message:
        '''
            Main method which will be called by the client code, as it is going to handle the entire system, i.e. core workflow logic
            Note conv_id will be retrieved by the main entry point where the text sent by user is being handled
        '''
        planner_input = self.build_planner_input(user_task = user_task, context = "", k_recent_messages = [], conversation_id = conversation_id, step_index = 1)

        planner_msg = self.planner.run(planner_input=planner_input)
        if planner_msg.status == 'failed' and planner_msg.target_agent is None:
            return planner_msg
        plan_response = planner_msg.response

        executor_msg = self.executor.run(conversation_id=conversation_id, step_index=2,plan_response=plan_response)
        if executor_msg.status == 'failed' and executor_msg.target_agent is None:
            print("Failed")
        return executor_msg
        
        #calling reviewer + its logic which should return if it accepts/rejects, then coord decides what to do w.r.t what it recieves

    def build_planner_input(self, user_task: str, context: str, k_recent_messages: list[dict], conversation_id: int, step_index: int ) -> dict:
        return {"task": user_task, "context": context, "k_recent_messages": k_recent_messages, "tools": self.planner_tool_descriptions, 
                "conversation_id": conversation_id, "step_index": step_index}
