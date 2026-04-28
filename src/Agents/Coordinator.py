from src.Agents.Planning.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent
from src.Agents.Routing.RouteAgent import RouteAgent
from src.message import Message
from src.OllamaClient import OllamaClient
import json

'''
    This is the agent which manages workflow, and configures the respective agents i.e. it acts like the runner method,
    and it tries to enforce certain heuristic methods which try to help the model be restricted in the way it can think (try to plan out something deterministicly
    this is because of hallucinations)
'''
class Coordinator():
    ROUTER_STEP = 1
    PLANNER_STEP = 2
    EXECUTOR_STEP = 3
    REVIEWER_STEP = 4
    MEMORY_STEP = 5
    FINAL_STEP = 6

    DIRECT_RESPONSE_TOOLS = {"direct_response"}

    READ_FILE_TOOLS = {
        "read_file", "find_file", "find_file_recursive",
        "list_dir", "list_tree", "search_text"
    }

    SUMMARISE_FILE_TOOLS = {
        "read_file", "find_file", "find_file_recursive", "summarise_txt"
    }

    MUTATION_FILE_TOOLS = {
        "generate_content", "create_file", "write_file", "append_file",
        "delete_file", "replace_text", "create_dir", "delete_dir",
        "move_path", "copy_path", "find_file", "find_file_recursive",
        "list_dir", "list_tree", "read_file", "summarise_txt", "search_text"
    }

    VALID_ROUTE_CONTRACTS = {
        "direct_response":      {"tool_group": "direct_response_tools",  "use_workspace": False},
        "memory_question":      {"tool_group": "direct_response_tools",  "use_workspace": False},
        "contextual_followup":  {"tool_group": "direct_response_tools",  "use_workspace": False},
        "workspace_read":       {"tool_group": "read_file_tools",        "use_workspace": True},
        "workspace_summarise":  {"tool_group": "summarise_file_tools",   "use_workspace": True},
        "workspace_mutation":   {"tool_group": "mutation_file_tools",    "use_workspace": True},
    }

    SAFE_FALLBACK_ROUTE = {
        "task_type": "direct_response",
        "tool_group": "direct_response_tools",
        "memory_mode": "pinned_only",
        "use_recent_messages": False,
        "use_workspace": False,
        "routing_reason": "Fallback applied because router output was invalid."
    }

    def __init__(self, planner: PlannerAgent, executor: ExecutorAgent, reviewer: ReviewerAgent,
                 memory: MemoryAgent, router: RouteAgent, planner_tool_descriptions: list[dict], tool_registry: dict, llm_client: OllamaClient):
        self.planner = planner
        self.planner_tool_descriptions = planner_tool_descriptions
        self.executor = executor
        self.tool_registry = tool_registry
        self.reviewer = reviewer
        self.memory = memory
        self.router = router
        self.llm_client = llm_client

    def build_planner_input(self, user_task: str, route: dict, conversation_id: int, step_index: int, replan_feedback: str = "") -> dict:
        scoped_context = self.build_scoped_context(user_task, route)
        if replan_feedback:
            scoped_context = f"{scoped_context}\n\n{replan_feedback}" if scoped_context else replan_feedback

        scoped_recent_messages = self.build_scoped_recent_messages(conversation_id, route)
        selected_tools = self.select_tools_for_route(route)
        scoped_workspace_contents = self.build_scoped_workspace_contents(route)

        return {
            "task": user_task,
            "context": scoped_context,
            "k_recent_messages": scoped_recent_messages,
            "tools": selected_tools,
            "workspace_contents": scoped_workspace_contents,
            "route": route,
            "conversation_id": conversation_id,
            "step_index": step_index,
        }

    def try_plan(self,planner_input: dict, max_attempts: int = 3) -> Message:
            '''
                Method which handles the raised exceptions for when the planner is running so that it manages the respective cases by replaning or 
                else tells coordinator that it can't handle this respective task.
            '''

            # Use a local copy so compiler-retry feedback does not permanently
            # modify the original planner_input used by other workflow stages.
            retry_planner_input = dict(planner_input)

            # Make sure context is copied as a string value.
            retry_planner_input["context"] = planner_input.get("context", "")

            for _ in range(max_attempts):
                planner_msg = self.planner.run(planner_input=retry_planner_input)
                self.memory.store_message(planner_msg)

                if planner_msg.status == "completed":
                    return planner_msg
                
                error = planner_msg.response.get("error", "")
                if "Unknown tool" in error:
                    return planner_msg   #stop immediately
                
                #Giving the planner retry some feedback about why the previous plan failed compilation
                retry_planner_input["context"] += f"""
                    Previous planning attempt failed during compilation.

                    Compiler error:
                    {error}

                    Fix only the structure of the plan.
                    The current task is still:
                    {retry_planner_input["task"]}

                    Important correction rule:
                    If a value comes from a previous step, use the argument name ending in _step.

                    Correct:
                    {{"content_step": 1}}
                    {{"text_step": 1}}
                    {{"path_step": 1}}

                    Wrong:
                    {{"content": "text_step:1"}}
                    {{"text": 1}}

                    For example:
                    If step 2 creates a file using the contents from step 1, use:
                    {{"path": "BabyClaw", "content_step": 1}}
                    """
            return planner_msg  #failed after retries

    

    def replan_after_review_rejection(self, conversation_id: int, user_task: str, plan_response: dict,
                                      executor_response: dict, reviewer_response: dict, approved_actions: set) -> Message:
        '''
            user_task stays pure. Rejection feedback goes into context only.
            Route is reused so replanning stays inside the same scoped environment.
        '''
        issues_text = "; ".join(reviewer_response.get("issues", [])) or "No specific issues provided."

        replan_feedback = (
            f"Previous attempt was rejected because: {issues_text}.\n"
            f"Plan again from scratch for the original user task. "
            f"Use *_step to chain values between steps. "
            f"Do not add steps unrelated to the original task. "
            f"Do not delete or read files the user did not mention."
        )

        route = plan_response.get("route") or self.run_router(conversation_id=conversation_id, user_task=user_task)
        route = self.validate_route(route)

        planner_input = self.build_planner_input(user_task=user_task, route=route, conversation_id=conversation_id, step_index=self.PLANNER_STEP,
                                                 replan_feedback=replan_feedback)

        planner_msg = self.try_plan(planner_input=planner_input)
        if planner_msg.status == "failed":
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Replanning failed after reviewer rejected the result.",
                                              issues=[planner_msg.response.get("error", "Unknown replanning error.")],
                                              execution_response=executor_response,
                                              error=planner_msg.response.get("error", "Unknown replanning error."),
                                              details=planner_msg.response)

        new_plan_response = planner_msg.response
        new_plan_response["route"] = route
        new_plan_response["workspace_before"] = self.tool_registry["list_tree"]["func"](path=".")

        execution_state = self.executor.initialise_execution_state(plan_response=new_plan_response,
                                                                    context=planner_input["context"],
                                                                    recent_messages=planner_input["k_recent_messages"],
                                                                    user_task=user_task)
        if approved_actions:
            execution_state["approved_actions"] = approved_actions

        executor_msg = self.run_execution_loop(conversation_id=conversation_id, execution_state=execution_state,
                                                start_step_index=self.EXECUTOR_STEP, user_task=user_task,
                                                plan_response=new_plan_response)

        if executor_msg.status == "waiting":
            return executor_msg

        if executor_msg.status == "failed":
            execution_state = executor_msg.response.get("execution_state", {})
            rollback_results = self.rollback_execution(executor_response=execution_state)
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Execution failed during replanning.",
                                              issues=[executor_msg.response.get("error", "Unknown execution error.")],
                                              execution_response=execution_state, rollback_results=rollback_results,
                                              error=executor_msg.response.get("error", "Unknown execution error."),
                                              details=executor_msg.response)

        return self.continue_workflow(conversation_id=conversation_id, user_task=user_task,
                                      plan_response=new_plan_response, executor_msg=executor_msg, review_retry_used=True)


    '''
        Messages sent to the runner file in order for the response field to be sent to the user in order for it to know what happened with its task
    '''
    def build_success_message(self, conversation_id: int, step_index: int, review_summary: str, execution_response: dict) -> Message:
        trace = execution_response.get("execution_trace", [])
        trace = sorted(trace, key=lambda step: step["id"])

        direct_outputs = [step.get("result") for step in trace if step.get("tool") == "direct_response" and step.get("status") == "completed" and step.get("result")]

        display_tools = {"read_file", "list_dir", "find_file", "summarise_txt", "create_file", "write_file", "append_file", "delete_file", 
                         "search_text", "replace_text","create_dir","delete_dir", "move_path", "copy_path","find_file_recursive", "list_tree"}

        display_outputs = [step.get("result") for step in trace if step.get("tool") in display_tools and step.get("status") == "completed" 
                           and step.get("result") is not None]
        
        display_result = direct_outputs[-1] if direct_outputs else (display_outputs[-1] if display_outputs else None)

        if display_result is not None:
            self.memory.store_message(Message(conversation_id=conversation_id, step_index=step_index, sender="assistant", receiver="user", 
                                            target_agent=None, message_type="assistant_message", status="completed", 
                                            response={"content": str(display_result)}, visibility="external"
                                            )
                                    )

        message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, 
                        message_type="workflow_result", status="completed",
                        response={
                                "message": "Task completed successfully.",
                                "review_summary": review_summary,
                                "execution_trace": trace,
                                "step_results": execution_response.get("step_results", {}),
                                "approved_actions": execution_response.get("approved_actions", []),
                                "rollback_log": execution_response.get("rollback_log", []),
                                "direct_response": direct_outputs[-1] if direct_outputs else None,
                                "display_result": display_result
                            },
                        visibility="external")
        
        self.memory.store_message(message)

        return message

    def build_failure_message(self, conversation_id: int, step_index: int, review_summary: str, issues: list[str],
                              execution_response: dict = None, rollback_results: list[dict] = None,
                              error: str = None, details: dict = None) -> Message:
        
        execution_response = execution_response or {}

        trace = execution_response.get("execution_trace", [])
        trace = sorted(trace, key=lambda step: step["id"]) if trace else []

        response = {
            "message": "Task could not be completed successfully.",
            "review_summary": review_summary,
            "issues": issues,
            "execution_trace": trace,
            "step_results": execution_response.get("step_results", {}),
            "approved_actions": execution_response.get("approved_actions", []),
            "rollback_log": execution_response.get("rollback_log", []),
            "rollback_results": rollback_results or []
        }

        if error:
            response["error"] = error

        if details:
            response["details"] = details

        message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, 
                          message_type="workflow_result", status="failed", response=response, visibility="external")
        
        self.memory.store_message(message)

        return message


    def start_workflow(self, conversation_id: int, user_task: str):
        user_message = Message(conversation_id=conversation_id, step_index=0, sender="user", receiver="coordinator",
                               target_agent=None, message_type="user_message", status="completed",
                               response={"content": user_task}, visibility="external")
        self.memory.store_message(message=user_message)

        route = self.run_router(conversation_id=conversation_id, user_task=user_task)

        planner_input = self.build_planner_input(user_task=user_task, route=route,
                                                 conversation_id=conversation_id, step_index=self.PLANNER_STEP)

        planner_msg = self.try_plan(planner_input=planner_input)

        if planner_msg.status == 'failed' and planner_msg.target_agent is None:
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Planning failed.",
                                              issues=[planner_msg.response.get("error", "Unknown planning error.")],
                                              error=planner_msg.response.get("error", "Unknown planning error."),
                                              details=planner_msg.response)

        plan_response = planner_msg.response
        plan_response["route"] = route
        plan_response["workspace_before"] = self.tool_registry["list_tree"]["func"](path=".")

        execution_state = self.executor.initialise_execution_state(plan_response=plan_response,
                                                                    context=planner_input["context"],
                                                                    recent_messages=planner_input["k_recent_messages"],
                                                                    user_task=user_task)
        executor_msg = self.run_execution_loop(conversation_id=conversation_id, execution_state=execution_state,
                                               start_step_index=self.EXECUTOR_STEP, user_task=user_task,
                                               plan_response=plan_response)

        if executor_msg.status == "waiting":
            return executor_msg

        if executor_msg.status == "failed":
            execution_state = executor_msg.response.get("execution_state", {})
            rollback_results = self.rollback_execution(executor_response=execution_state)
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Execution failed before review.",
                                              issues=[executor_msg.response.get("error", "Unknown execution error.")],
                                              execution_response=execution_state, rollback_results=rollback_results,
                                              error=executor_msg.response.get("error", "Unknown execution error."),
                                              details=executor_msg.response)

        return self.continue_workflow(conversation_id=conversation_id, user_task=user_task,
                                      plan_response=plan_response, executor_msg=executor_msg, review_retry_used=False)
    
    def continue_workflow(self, conversation_id: int, user_task: str, plan_response: dict, executor_msg: Message, review_retry_used: bool) -> Message:
        '''
            This method is handling the respective logic which happens after permission is accepted (if needed), so that the user is kept in the loop while
            also not having redundant lines of code in the coordinator
        '''

        executor_response = executor_msg.response #will only contain messages with .status == "completed"

        #Reviewer needs to see the actual workspace state to verify set-based claims
        #like "delete all files except X" — the trace alone is insufficient.
        list_tree = self.tool_registry.get("list_tree", {}).get("func")
        workspace_after = list_tree(path=".") if list_tree else []
        workspace_before = executor_response.get("workspace_before", [])

        reviewer_msg = self.reviewer.run(conversation_id=conversation_id, step_index=self.REVIEWER_STEP, user_task=user_task, 
                                         execution_response=executor_response, workspace_before=workspace_before, workspace_after=workspace_after)
        self.memory.store_message(message=reviewer_msg)

        reviewer_response = reviewer_msg.response
        accepted = reviewer_msg.response.get("accepted", False)

        if not accepted:
            #Reviewer rejected the work, so any side-effects (created/modified/deleted files)
            #must be undone here. Previously rollback_results was hardcoded to [] and the
            #workspace was left dirty after a rejection.
            rollback_results = self.rollback_execution(executor_response=executor_response)

            issues = reviewer_response.get("issues", [])
            self.memory.store_message(message=Message(conversation_id=reviewer_msg.conversation_id, step_index=reviewer_msg.step_index, 
                                                    sender="memory", receiver="coordinator", target_agent=None, message_type="memory_store", 
                                                    status="completed", response={"stored": False, "reason": "Reviewer did not accept workflow"},
                                                    visibility="internal"))

            if not review_retry_used:
                approved_actions = set(executor_response.get("approved_actions", []))
                return self.replan_after_review_rejection(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response,
                                                        executor_response=executor_response, reviewer_response=reviewer_response, approved_actions=approved_actions)

            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP, 
                                            review_summary=reviewer_response.get("review_summary", "Reviewer rejected execution."),
                                            issues=issues,execution_response=executor_response, rollback_results=rollback_results, details=reviewer_response)

        memory_msg = self.memory.store_long_term_memory(user_task=user_task, conversation_id=conversation_id, step_index=self.MEMORY_STEP)
        self.memory.store_message(message=memory_msg)

        return self.build_success_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                          review_summary=reviewer_response.get("review_summary", "Execution accepted."), 
                                          execution_response=executor_response)

    
    def run_execution_loop(self, conversation_id: int, execution_state: dict, start_step_index: int, user_task: str, plan_response: dict) -> Message:
        '''
            Interleaving with the executor agent in order to tell it when to run, since certain tasks would need permission before it does so.
        '''

        #Initialising it to this respective message if while loop does not re-set execution_msg to the corresponding message
        executor_msg = Message(conversation_id=conversation_id, step_index=start_step_index, sender="coordinator", receiver="user", target_agent=None, 
                               message_type="workflow_result",status="failed", 
                               response={"message": "Planner produced no executable steps."}, visibility="external")
        
        while not self.executor.is_execution_complete(execution_state):
            runnable_steps = self.executor.get_runnable_steps(execution_state)

            if not runnable_steps:
                #since while loop is confirming that the execution is not yet completete then runnable_steps has to be with some respective steps, hence if it is empty -> failure
                return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                                  review_summary="Execution blocked: no runnable steps found.",
                                                  issues=["Execution is not complete, but no runnable steps are available."],
                                                  execution_response=execution_state,
                                                  error="Execution blocked: no runnable steps found.",
                                                  details=execution_state)

            try:
                permission_steps = self.get_permission_required_steps(runnable_steps, execution_state=execution_state)
            except Exception as e:
                return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                                  review_summary="Execution failed while resolving permission arguments.", issues=[str(e)],
                                                  execution_response=execution_state, error=str(e),details={"runnable_steps": runnable_steps}
                                                  )

            if permission_steps:
                #Exits back to the start_workflow method which has to be routed back to the user so that it asks it the respective message being sent
                return self.build_permission_request_message(conversation_id=conversation_id, step_index=start_step_index, user_task=user_task, plan_response=plan_response, 
                                                             permission_steps=permission_steps,execution_state=execution_state, pending_runnable_steps=runnable_steps)

            executor_msg = self.executor.run_steps(conversation_id=conversation_id, step_index=start_step_index,execution_state=execution_state,runnable_steps=runnable_steps)
            self.memory.store_message(executor_msg)

            if executor_msg.status == "failed":
                return executor_msg
            
            if executor_msg.message_type == "execution_wave_result":
                execution_state = executor_msg.response
                continue

        return executor_msg
    
    def continue_after_permission(self, conversation_id: int, user_task: str, plan_response: dict, execution_state: dict,
                                  pending_runnable_steps: list[dict], step_index: int, approved: bool) -> Message:
        '''
            This method is called by the runner after the user replies to the permission request.
            If permission is denied then the workflow stops, otherwise the saved runnable wave is executed and the workflow continues.

            Note pending_runnable_steps would be set by the requested tools which is built inside the build_permission_request_message and execution_state would
            be set from that respective message sent by that function
        '''

        if not approved:
            message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, 
                           message_type="workflow_result", status="cancelled",response={"message": "Noted. The task will not be executed."}, visibility="external")
            
            self.memory.store_message(message)
            return message
        
        approved_action_signatures = []

        for step in pending_runnable_steps:
            if not self.tool_registry.get(step["tool"], {}).get("requires_permission", False):
                continue

            resolved_args = self.executor.resolve_step_args_for_permission(step=step, execution_state=execution_state)

            approved_action_signatures.append(self.permission_signature(step, resolved_args))

        execution_state["approved_step_ids"].update(step["id"] for step in pending_runnable_steps 
                                                    if self.tool_registry.get(step["tool"], {}).get("requires_permission", False)
                                                    )

        execution_state.setdefault("approved_actions", set()).update(approved_action_signatures)

        #executing the exact runnable wave which was already shown to the user and approved
        executor_msg = self.executor.run_steps(conversation_id=conversation_id, step_index=step_index, execution_state=execution_state, 
                                                   runnable_steps=pending_runnable_steps)
        self.memory.store_message(message=executor_msg)

        #if execution finished right after the approved wave then continue to reviewer/memory/final response
        if executor_msg.message_type == "execution_result":
            return self.continue_workflow(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response, 
                              executor_msg=executor_msg, review_retry_used=False)
        
        #Handling the case that if when the set of permitted runnable tools fail when executing it stops
        if executor_msg.status == "failed":
            execution_state = executor_msg.response.get("execution_state", {})
            rollback_results = self.rollback_execution(executor_response=execution_state)

            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Execution failed after permission was approved.",
                                              issues=[executor_msg.response.get("error", "Unknown execution error.")],
                                              execution_response=execution_state,
                                              rollback_results=rollback_results,
                                              error=executor_msg.response.get("error", "Unknown execution error."),
                                              details=executor_msg.response)

        #if the approved wave completed but there are still remaining steps then continue the execution loop normally
        if executor_msg.message_type == "execution_wave_result":
            updated_execution_state = executor_msg.response

            next_executor_msg = self.run_execution_loop(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response, 
                                                        execution_state=updated_execution_state, start_step_index=step_index)

            if next_executor_msg.status == "waiting":
                return next_executor_msg #will re-rout the respective permission request back to user again
            
            if next_executor_msg.status=="failed":
                execution_state = next_executor_msg.response.get("execution_state", {})
                rollback_results = self.rollback_execution(executor_response=execution_state)

                return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                                  review_summary="Execution failed while continuing after permission.",
                                                  issues=[next_executor_msg.response.get("error", "Unknown execution error.")],
                                                  execution_response=execution_state, rollback_results=rollback_results,
                                                  error=next_executor_msg.response.get("error", "Unknown execution error."),
                                                  details=next_executor_msg.response)

            return self.continue_workflow(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response, 
                                          executor_msg=next_executor_msg, review_retry_used=False)
        
        #Fall back return statement if something unexpected breaks inside the workflow
        return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                          review_summary="Unexpected breakage in the workflow.",
                                          issues=["The executor returned an unexpected message type."],
                                          error="Unexpected breakage in the workflow.",
                                          details=executor_msg.response)
    
    
    def permission_signature(self, step: dict, resolved_args: dict) -> str:
        tool_name = step["tool"]

        tool_def = self.tool_registry.get(tool_name, {})
        identity_args = tool_def.get("permission_identity_args") or []

        if identity_args:
            identity = {
                arg_name: resolved_args.get(arg_name)
                for arg_name in identity_args
            }
        else:
            identity = resolved_args

        return json.dumps(
            {
                "tool": tool_name,
                "identity": identity
            },
            sort_keys=True
        )
    
    def get_permission_required_steps(self, runnable_steps: list[dict], execution_state: dict) -> list[dict]:
        permission_steps = []

        approved_ids = execution_state.get("approved_step_ids", set())
        approved_actions = execution_state.get("approved_actions", set())

        for step in runnable_steps:
            tool_name = step["tool"]
            tool_def = self.tool_registry.get(tool_name, {})

            if not tool_def.get("requires_permission", False):
                continue

            if step["id"] in approved_ids:
                continue

            resolved_args = self.executor.resolve_step_args_for_permission(
                step=step,
                execution_state=execution_state
            )

            signature = self.permission_signature(step, resolved_args)

            if signature in approved_actions:
                continue

            permission_steps.append(step)

        return permission_steps #represents another sublist but this time of the respective list which contains the steps that are currently available to be executed
    
    def build_permission_request_message(self, conversation_id: int, step_index: int, user_task: str, plan_response: dict, permission_steps: list[dict], 
                                        execution_state: dict, pending_runnable_steps: list[dict]) -> Message:
        requested_tools = []

        for step in permission_steps:
            tool_name = step["tool"]
            tool_def = self.tool_registry.get(tool_name, {})
            description = tool_def.get("description", "No description available.")
            requested_tools.append({
                "step_id": step["id"],
                "tool": tool_name,
                "args": step["args"],
                "description": description
            })

        message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, message_type="permission_request", status="waiting",
                    response={
                        "message": "Do you approve of these requested tools to be executed?",
                        "requested_tools": requested_tools,
                        "user_task": user_task,
                        "plan_response": plan_response,
                        "execution_state": execution_state,
                        "pending_runnable_steps": pending_runnable_steps,
                        "next_step_index": step_index
                    },
                        visibility="external")
        
        self.memory.store_message(message)

        return message
    
    def rollback_execution(self, executor_response: dict) -> list[dict]:
        rollback_results = []

        rollback_log = executor_response.get("rollback_log", [])

        for entry in reversed(rollback_log):
            tool_name = entry["tool"]
            tool_def = self.tool_registry.get(tool_name, {})
            rollback_apply = tool_def.get("rollback_apply")

            if not rollback_apply:
                continue

            try:
                rollback_apply(entry["snapshot"])
                rollback_results.append({
                    "step_id": entry["step_id"],
                    "tool": tool_name,
                    "status": "rolled_back"
                })
            except Exception as e:
                rollback_results.append({
                    "step_id": entry["step_id"],
                    "tool": tool_name,
                    "status": "rollback_failed",
                    "error": str(e)
                })

        return rollback_results
    
    def validate_route(self, route: dict) -> dict:
        '''
            Enforces deterministic route contracts. If router output violates a contract, fall back to safe defaults.
        '''
        if not isinstance(route, dict) or "task_type" not in route:
            return dict(self.SAFE_FALLBACK_ROUTE)

        task_type = route.get("task_type")
        contract = self.VALID_ROUTE_CONTRACTS.get(task_type)

        if contract is None:
            return dict(self.SAFE_FALLBACK_ROUTE)

        if route.get("tool_group") != contract["tool_group"]:
            return dict(self.SAFE_FALLBACK_ROUTE)

        if route.get("use_workspace") != contract["use_workspace"]:
            return dict(self.SAFE_FALLBACK_ROUTE)

        if route.get("memory_mode") not in {"none", "pinned_only", "relevant_only", "full"}:
            return dict(self.SAFE_FALLBACK_ROUTE)

        if not isinstance(route.get("use_recent_messages"), bool):
            return dict(self.SAFE_FALLBACK_ROUTE)

        return route

    def select_tools_for_route(self, route: dict) -> list[dict]:
        tool_group = route["tool_group"]

        if tool_group == "direct_response_tools":
            allowed = self.DIRECT_RESPONSE_TOOLS
        elif tool_group == "read_file_tools":
            allowed = self.READ_FILE_TOOLS
        elif tool_group == "summarise_file_tools":
            allowed = self.SUMMARISE_FILE_TOOLS
        elif tool_group == "mutation_file_tools":
            allowed = self.MUTATION_FILE_TOOLS
        else:
            allowed = self.DIRECT_RESPONSE_TOOLS

        return [t for t in self.planner_tool_descriptions if t["name"] in allowed]

    def build_scoped_context(self, user_task: str, route: dict) -> str:
        return self.memory.get_memory_by_mode(task=user_task, mode=route["memory_mode"], k=5)

    def build_scoped_recent_messages(self, conversation_id: int, route: dict) -> list[dict]:
        if not route["use_recent_messages"]:
            return []
        return self.memory.get_recent_conversation_messages(conversation_id=conversation_id, k=5)

    def build_scoped_workspace_contents(self, route: dict):
        if not route["use_workspace"]:
            return []
        return self.tool_registry["list_dir"]["func"](path=".")

    def run_router(self, conversation_id: int, user_task: str) -> dict:
        route_msg = self.router.run(conversation_id=conversation_id, step_index=self.ROUTER_STEP, user_task=user_task)
        self.memory.store_message(route_msg)

        if route_msg.status != "completed":
            return dict(self.SAFE_FALLBACK_ROUTE)

        return self.validate_route(route_msg.response)