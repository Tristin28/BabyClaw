from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message 
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class ExecutorAgent(Agent):
    def __init__(self, llm_client:OllamaClient, tool_registry: dict):
        self.llm_client = llm_client
        super().__init__("executor")
        self.tool_registry = tool_registry #setting as an instance field so that sets of tools can be exchanged depending on what user enables, i.e. coord sets it

    def run(self, conversation_id: int, step_index: int, plan_response: dict) -> Message:
        try:
            execution_response = self.execute_plan(plan_response)
            status = "completed"
        except Exception as e:
            execution_response = {"error": str(e)}
            status = "failed"

        return self.get_results(conversation_id=conversation_id, step_index=step_index, execution_response=execution_response, status=status)

    def get_results(self, conversation_id: int, step_index: int, execution_response: dict, status: str) -> Message:
        '''
            Wrapper method which is wrapping the execution response into a Message DTO so it can be sent to the coordinator.
        '''
        
        target_agent = None
        if status == "completed":
            target_agent = "reviewer"

        return self.get_message(conversation_id=conversation_id, step_index=step_index, receiver="coordinator", target_agent=target_agent,message_type="execution_result",
                                status=status, response=execution_response, visibility="internal"
                                )

    def validate_result(self, tool_name: str, result: Any):
        '''
            Validates whether the result (being a built in python object) returned by a tool is usable 
            (only checking built in objects as other objects can have diff semantics, as then semantics will be checked by the reviewer agent)
        '''
        if result is None:
            raise ValueError(f"Tool '{tool_name}' returned None")

        if isinstance(result, str) and result.strip() == "":
            raise ValueError(f"Tool '{tool_name}' returned an empty string")

        if isinstance(result, (list, dict, tuple, set)) and len(result) == 0:
            raise ValueError(f"Tool '{tool_name}' returned an empty {type(result).__name__}")

    def execute_plan(self, plan_response: dict) -> dict:
        step_results = {} #stores the results of each step_id i.e. step_id: result
        step_status = {} #stores the step_id as a key and each value is either 'completed' or 'failed'
        remaining_steps = plan_response["steps"][:] #Copying the list so that we dont remove elements from the actual step list too (since mutable)

        execution_trace = []

        while remaining_steps:
            runnable_steps = self.get_runnable_steps(remaining_steps, step_status)
            if not runnable_steps: 
                raise RuntimeError("Execution blocked: no runnable steps found")
            
            #Deciding how many threads should be created inside of the thread pool so there is no wastage of resources
            MAX_THREADS = 8
            max_workers = max(1, min(len(runnable_steps), MAX_THREADS)) 

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                #Dict where key = result (of type future) and value would be current step being executed (i.e. value = dict)
                future_step_result = {executor.submit(self.execute_step,step, step_results): step for  step in runnable_steps}

                completed_ids = []

                for future in as_completed(future_step_result):
                    step = future_step_result[future]

                    try:
                        result = future.result()

                        step_results[step["id"]] = result
                        step_status[step["id"]] = "completed"
                        completed_ids.append(step["id"])

                        execution_trace.append({"id": step["id"], "tool": step["tool"], "status": "completed", "result": result})

                    except Exception as e:
                        step_status[step["id"]] = "failed"
                        execution_trace.append({"id": step["id"], "tool": step["tool"], "status": "failed", "error": str(e)})
                        
                        #propogating the error upward into the stack frames which will be handled by the run method however, it has to be directed accordingly in coord
                        raise 

            remaining_steps = [step for step in remaining_steps if step["id"] not in completed_ids]#Removing the steps which are completed
        
        return {"goal": plan_response["goal"], "step_results": execution_trace}


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
            if self.dependencies_satisified(step, step_status):
                runnable.append(step)

        return runnable

    def dependencies_satisified(self, step: dict, step_status: dict) -> bool:
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