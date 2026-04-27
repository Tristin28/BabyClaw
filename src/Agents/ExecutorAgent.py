from src.Agents.BaseAgent import Agent
from src.message import Message 
from typing import Any
#from concurrent.futures import ThreadPoolExecutor, as_completed - removed parallel work

class ExecutorAgent(Agent):
    def __init__(self, tool_registry: dict):
        super().__init__("executor")
        self.tool_registry = tool_registry #setting as an instance field so that sets of tools can be exchanged depending on what user enables, i.e. coord sets it

    def initialise_execution_state(self, plan_response: dict, context: str = "", recent_messages: list[dict] = None, user_task: str = "") -> dict:
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
            "execution_trace": [],
            "approved_step_ids": set(), #storing which steps already got permission - in order to fix the bug which was asking permission many times for the same thing
            "approved_actions": set(),
            "rollback_log": [],
            "context": context,
            "recent_messages": recent_messages or [], #Depending on whether it is falsy or not
            "user_task": user_task, #pinned literal user task; executor injects it into LLM prompts to stop the planner from drifting
            "workspace_before": plan_response.get("workspace_before", [])
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
            "execution_trace": execution_state["execution_trace"],
            "step_results": execution_state["step_results"],
            "approved_actions": list(execution_state.get("approved_actions", set())),
            "rollback_log": execution_state.get("rollback_log", []),
            "workspace_before": execution_state.get("workspace_before", [])
        }
    
    def validate_result(self, tool_name: str, result: Any):
        '''
            Validates whether the result returned by a tool is usable.
            This should be tool-specific because some empty results are valid.
        '''
        if result is None:
            raise ValueError(f"Tool '{tool_name}' returned None")

        NON_EMPTY_STRING_TOOLS = { "direct_response", "summarise_txt", "create_file", "write_file", "append_file", "find_file"}

        if tool_name in NON_EMPTY_STRING_TOOLS:
            if not isinstance(result, str) or result.strip() == "":
                raise ValueError(f"Tool '{tool_name}' returned an empty string")
            
    def execute_tools(self, execution_state: dict, runnable_steps: list[dict]) -> dict:
        #Making use of mutability, as dict object is mutable then if some changes happen by any reference all other pointers pointing to respective obj will see the change
        step_results = execution_state["step_results"] #stores the results of each step_id i.e. step_id: result
        step_status = execution_state["step_status"] #stores the step_id as a key and each value is either 'completed' or 'failed'

        execution_trace = execution_state["execution_trace"]

        completed_ids = []

        for step in runnable_steps:
            resolved_args = {}

            try:
                result, resolved_args = self.execute_step(step, execution_state)
                self.validate_result(step["tool"], result)

                step_results[step["id"]] = result
                step_status[step["id"]] = "completed"
                completed_ids.append(step["id"])

                execution_trace.append({
                    "id": step["id"],
                    "tool": step["tool"],
                    "args": step.get("args", {}),
                    "resolved_args": resolved_args,
                    "depends_on": step.get("depends_on", []),
                    "status": "completed",
                    "result": result
                })

            except Exception as e:
                step_status[step["id"]] = "failed"

                execution_trace.append({
                    "id": step["id"],
                    "tool": step["tool"],
                    "args": step.get("args", {}),
                    "resolved_args": resolved_args,
                    "depends_on": step.get("depends_on", []),
                    "status": "failed",
                    "error": str(e)
                })
                
                #propogating the error upward into the stack frames which will be handled by the run method however, it has to be directed accordingly in coord
                raise RuntimeError(f"Step {step['id']} failed: {e}") from e

        #Removing the steps which are completed
        execution_state["remaining_steps"] = [step for step in execution_state["remaining_steps"] if step["id"] not in completed_ids] 

        return execution_state
    
    def execute_step(self, plan_output_step: dict, execution_state: dict) -> tuple[Any, dict]:
        tool_name = plan_output_step["tool"]

        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool {tool_name}")
        
        tool_def = self.tool_registry[tool_name]

        tool_fn = tool_def["func"] #stores the actual method
        input_map = tool_def["input_map"] #stores the arg name as key and its value would be key inside args (dict inside plan_output_step)

        resolved_args = self.resolve_args(
            plan_output_step=plan_output_step,
            execution_state=execution_state,
            input_map=input_map
        ) 

        snapshot = None
        rollback_snapshot_fn = tool_def.get("rollback_snapshot")

        #Taking the rollback snapshot before the tool runs, because this stores the old file state before it gets modified
        if rollback_snapshot_fn:
            snapshot = rollback_snapshot_fn(**resolved_args)

            execution_state.setdefault("rollback_log", []).append({
                "step_id": plan_output_step["id"],
                "tool": tool_name,
                "resolved_args": resolved_args,
                "snapshot": snapshot
            })

        result = tool_fn(**resolved_args)

        return result, resolved_args
        
    def resolve_args(self, plan_output_step: dict, execution_state: dict, input_map: dict) -> dict: 
        '''
                Takes in the argument values selected by the llm in the planning stage and initialisign them to the respective local variables of the chosen funciton
                Where input_map will describe the argument name and what its value should be, and args will contain the value placeholder inside of input map as the key
                and its value would be the value which needs to be set to that argument, and if this placeholder name would be source_step it indicates that the argument
                needs to be initialised from a previous step
        '''
        tool_name = plan_output_step["tool"] 
        args = plan_output_step.get("args", {})
        step_results = execution_state["step_results"]

        resolved_args = {}
        for real_arg, planner_arg in input_map.items():

            if planner_arg in args:
                resolved_args[real_arg] = args[planner_arg]
                continue

            step_arg = f"{planner_arg}_step"

            if step_arg in args:
                step_id = args[step_arg]

                if not isinstance(step_id, int):
                    raise ValueError(f"{step_arg} must reference an integer step id")

                if step_id not in step_results:
                    raise ValueError(f"Step result {step_id} not available yet")

                resolved_args[real_arg] = step_results[step_id]
                continue

            raise ValueError(f"Missing planner argument: expected '{planner_arg}' or '{step_arg}'")

        if tool_name == "direct_response":
            resolved_args["context"] = execution_state.get("context", "")
            resolved_args["recent_messages"] = execution_state.get("recent_messages", [])

            #Force the user's actual task back into the prompt so the planner cannot
            #silently swap the request (e.g. "snake game" -> "random numbers").
            user_task = execution_state.get("user_task", "")
            planner_prompt = resolved_args.get("prompt", "")
            if user_task:
                resolved_args["prompt"] = (
                    f"User task (authoritative):\n{user_task}\n\n"
                    f"Planner instruction:\n{planner_prompt}"
                )

        if tool_name == "generate_content":
            user_task = execution_state.get("user_task", "")
            planner_prompt = resolved_args.get("prompt", "")
            if user_task:
                resolved_args["prompt"] = (
                    f"User task (authoritative):\n{user_task}\n\n"
                    f"Planner instruction (use only to refine, do not contradict the user task):\n{planner_prompt}"
                )

        return resolved_args
    
    def resolve_step_args_for_permission(self, step: dict, execution_state: dict) -> dict:
        '''
            It is small wrapper method which is used by Coordinator before permission checking
        '''
        tool_name = step["tool"]

        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool {tool_name}")

        tool_def = self.tool_registry[tool_name]
        input_map = tool_def["input_map"]

        return self.resolve_args(
            plan_output_step=step,
            execution_state=execution_state,
            input_map=input_map
        )

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
                message_type = "execution_result" 
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