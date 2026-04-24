from src.Agents.BaseAgent import Agent
from src.OllamaClient import OllamaClient
from src.message import Message 
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class ExecutorAgent(Agent):
    def __init__(self, tool_registry: dict):
        super().__init__("executor")
        self.tool_registry = tool_registry #setting as an instance field so that sets of tools can be exchanged depending on what user enables, i.e. coord sets it

    def initialise_execution_state(self, plan_response: dict) -> dict:
        ''' 
            Initialisting an execution state so that the coordinator would be the one which keeps track of the steps running,
            this is done so that it interleaves with the executor and if any runnable steps are to have permission it asks the user before it executes the
            respective tools.
        '''
        return {
            "goal": plan_response["goal"],
            "remaining_steps": plan_response["steps"][:], #Copying the list so that we dont remove elements from the actual step list too (since mutable)
            "step_results": {},
            "step_status": {},
            "execution_trace": []
        }
    
    def is_execution_complete(self, execution_state: dict) -> bool:
        return len(execution_state["remaining_steps"]) == 0
    
    def get_runnable_wave(self, execution_state: dict) -> list[dict]:
        remaining_steps = execution_state["remaining_steps"]
        step_status = execution_state["step_status"]
        return self.get_runnable_steps(remaining_steps, step_status) #will retutrn the respective sublist containing only dict items of which steps are valid to be executed
    
    def get_runnable_steps(self, remaining_steps, step_status) -> list[dict]:
        runnable = []

        for step in remaining_steps:
            if self.dependencies_satisfied(step, step_status):
                runnable.append(step)

        return runnable #would represent a sublist of the actual list i.e. current step items to be executed

    def dependencies_satisfied(self, step: dict, step_status: dict) -> bool:
        for dep_id in step.get("depends_on",[]):
            if step_status.get(dep_id) != "completed":
                return False
        return True
    
    def build_execution_result(self, execution_state: dict) -> dict:
        return {
            "goal": execution_state["goal"],
            "step_results": execution_state["execution_trace"]
        }
    
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

    def execute_tools(self, execution_state: dict, runnable_steps: list[dict]) -> dict:
        #Making use of mutability, as dict object is mutable then if some changes happen by any reference all other pointers pointing to respective obj will see the change
        step_results = execution_state["step_results"] #stores the results of each step_id i.e. step_id: result
        step_status = execution_state["step_status"] #stores the step_id as a key and each value is either 'completed' or 'failed'

        execution_trace = execution_state["execution_trace"]

       
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
                    self.validate_result(step["tool"], result)

                    step_results[step["id"]] = result
                    step_status[step["id"]] = "completed"
                    completed_ids.append(step["id"])

                    execution_trace.append({"id": step["id"], "tool": step["tool"], "status": "completed", "result": result})

                except Exception as e:
                    step_status[step["id"]] = "failed"
                    execution_trace.append({"id": step["id"], "tool": step["tool"], "status": "failed", "error": str(e)})
                    
                    #propogating the error upward into the stack frames which will be handled by the run method however, it has to be directed accordingly in coord
                    raise RuntimeError(f"Step {step['id']} failed: {e}") from e

            #Removing the steps which are completed
            execution_state["remaining_steps"] = [step for step in execution_state["remaining_steps"] if step["id"] not in completed_ids] 
         
        return execution_state

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

    def run_set_tools(self, conversation_id: int, step_index: int, execution_state: dict, runnable_steps: list[dict]) -> Message:
        '''
            Wrapper function for execute_tools so that it calls it and then the respective execution state it returns is then checked whether 
            it is entirely complete, not yet complete or something failed
        '''
        try:
            updated_execution_state = self.execute_tools(execution_state=execution_state, runnable_steps=runnable_steps)
            if self.is_execution_complete(updated_execution_state):
                execution_response = self.build_execution_result(updated_execution_state)
                target_agent = "reviewer"
                message_type = "last batch of steps are completed" 
            else:
                target_agent = None
                message_type = "execution_wave_result" 
                execution_response = updated_execution_state

            status = "completed"
        except Exception as e:
            status = "failed"
            target_agent = None
            message_type = "execution_failed"
            execution_response = {
                "error": str(e),
                "execution_state": execution_state
             }

        return self.get_message(conversation_id=conversation_id, step_index=step_index, receiver="coordinator", target_agent=target_agent,message_type=message_type,
                            status=status, response=execution_response, visibility="internal"
                            )