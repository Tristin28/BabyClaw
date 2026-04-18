from src.Agents.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.ReviewerAgent import ReviewerAgent

class Coordinator():
    def __init__(self,planner:PlannerAgent, executor:ExecutorAgent, reviewer:ReviewerAgent, memory:MemoryAgent):
        self.planner = planner
        self.executor = executor
        self.reviewer = reviewer
        self.memory = memory
        self.step_index = 0 