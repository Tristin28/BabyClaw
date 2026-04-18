from Agents.PlannerAgent import PlannerAgent
from Agents.ExecutorAgent import ExecutorAgent
from Agents.MemoryAgent import MemoryAgent
from Agents.ReviewerAgent import ReviewerAgent

class Coordinator():
    def __init__(self,planner:PlannerAgent, executor:ExecutorAgent, reviewer:ReviewerAgent, memory:MemoryAgent):
        self.planner = planner
        self.executor = executor
        self.reviewer = reviewer
        self.memory = memory
        self.step_index = 0 