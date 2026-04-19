from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message 
from typing import Any

class ExecutorAgent(Agent):
    def __init__(self, llm_client:OllamaClient, tool_registry: dict):
        self.llm_client = llm_client
        super().__init__("executor")
        self.tool_registry = tool_registry #setting as an instance field so that sets of tools can be exchanged depending on what user enables, i.e. coord sets it

    def get_results(self) -> Message:
        pass

    def validate_result(self):
        pass

    def execute_plan(self, plan_response: dict):
        step_results = {} #stores the results of each step_id
        step_status = {} #stores the step_id as a key and each value is either 'completed' or 'failed'
        remaining_steps = plan_response["steps"][:] #Copying the list so that we dont remove elements from the actual step list too (since mutable)

        while remaining_steps:
            runnable_steps = self.get_runnable_steps(remaining_steps, step_status)

            for step in runnable_steps:
                result = self.execute_step(step, step_results)

                step_results[step["id"]] = result
                step_status[step["id"]] = "completed"

                remaining_steps.remove(step)


    def execute_step(self, plan_output_step: dict, step_results: dict) -> Any:
        tool_name = plan_output_step["tool"]
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool {tool_name}")
        
        tool_def = self.tool_registry[tool_name]

        tool_fn = tool_def ["func"] #stores the actual method
        input_map = tool_def["input_map"] #stores the arg name as key and its value would be key inside args (dict inside plan_output_step)

        resolved_args = self.resolve_args(step_results=step_results, input_map=input_map, args =plan_output_step["args"] )

        result = tool_fn(**resolved_args)
        return result
    
    def get_runnable_steps(self, remaining_steps, step_status) -> list[dict]:
        runnable = []

        for step in remaining_steps:

            deps = step["depends_on"]

            if all(step_status.get(dep) == "completed" for dep in deps):
                runnable.append(step)

        return runnable

    def dependencies_satisifed(self, step: dict, step_status: dict) -> bool:
        for dep_id in step["depends_on"]:
            if step_status.get(dep_id) != "completed":
                return False
        return True
    
    def resolve_args(self, step_results: dict, input_map: dict, args: dict) -> dict:
        '''
                Takes in the argument values selected by the llm in the planning stage and initialisign them to the respective local variables of the chosen funciton
                Where input_map will describe the argument name and what its value should be, and args will contain the value placeholder inside of input map as the key
                and its value would be the value which needs to be set to that argument, and if this placeholder name would be source_step it indicates that the argument
                needs to be initialised from a previous step
        '''
        resolved_args = {}
        for real_arg, planner_arg in input_map.items():
            if planner_arg not in args:
                raise ValueError(f"Missing planner argument: '{planner_arg}'")
            
            value = args[planner_arg]

            if planner_arg.endswith("_step"):
                if not isinstance(value, int):
                    raise ValueError(f"{planner_arg} must reference step id")

                if value not in step_results:
                    raise ValueError(f"Step result {value} not available yet")
                
                resolved_args[real_arg] = step_results[value]
            else:
                resolved_args[real_arg] = value
        
        return resolved_args