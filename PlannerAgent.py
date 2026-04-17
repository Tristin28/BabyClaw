from OllamaClient import OllamaClient

class PlannerAgent():
    def __init__(self, llm_client:OllamaClient):
        self.llm_client = llm_client

    def create_plan(self, task:str):
        prompt = f" "
