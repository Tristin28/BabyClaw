from src.Agents.Planning.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent
from src.message import Message
from src.OllamaClient import OllamaClient
import json


'''
    This is the agent which manages workflow, and configures the respective agents i.e. it acts like the runner method,
    and it tries to enforce certain heuristic methods which try to help the model be restricted in the way it can think (try to plan out something deterministicly
    this is because of hallucinations)
'''
class Coordinator():
    PLANNER_STEP = 1
    EXECUTOR_STEP = 2
    REVIEWER_STEP = 3
    MEMORY_STEP = 4
    FINAL_STEP = 5

    #Tools whose path argument should be pinned to the user-provided filename.
    PATH_PINNED_TOOLS = {"create_file", "write_file", "append_file", "delete_file"}

    #Phrases that signal a file extension when the user did not type one.
    MODE_EXTENSIONS = {
        "text mode": "txt",
        "plain text": "txt",
        "as text": "txt",
        "txt format": "txt",
        "python": "py",
        "py mode": "py",
        "markdown": "md",
        "md mode": "md",
        "json": "json",
        "csv": "csv",
    }

    def __init__(self,planner:PlannerAgent, executor:ExecutorAgent, reviewer:ReviewerAgent, memory:MemoryAgent, 
                 planner_tool_descriptions: list[dict], tool_registry: dict, llm_client: OllamaClient):
        self.planner = planner
        self.planner_tool_descriptions = planner_tool_descriptions
        self.executor = executor
        self.tool_registry = tool_registry
        self.reviewer = reviewer
        self.memory = memory
        self.llm_client = llm_client

    def build_planner_input(self, user_task: str, context: str, k_recent_messages: list[dict], conversation_id: int, step_index: int ) -> dict:
        '''
            Building the respective planner input for the planner agent to handle it and process it, so that the llm would know how to reason
        '''
        workspace_contents = self.tool_registry["list_dir"]["func"](path=".") #giving the planner llm access to what contents the directory it operates over contains

        # Important:
        # user_task must always be the original current user task as a string.
        # Context, recent messages, and workspace contents are only extra support.
        is_workspace_task = self.is_likely_workspace_task(user_task)

        available_tools = self.planner_tool_descriptions

        if not is_workspace_task:
            available_tools = [
                tool for tool in self.planner_tool_descriptions
                if tool["name"] == "direct_response"
            ]

        print("\n[DEBUG - PLANNER INPUT]")
        print("user_task:", user_task)
        print("is_likely_workspace_task:", is_workspace_task)
        print("available_tools:", [tool["name"] for tool in available_tools])
        print("context:", context)
        print("[/DEBUG - PLANNER INPUT]\n")

        return {
            "task": user_task,
            "context": context,
            "k_recent_messages": k_recent_messages,
            "tools": available_tools,
            "conversation_id": conversation_id,
            "step_index": step_index,
            "workspace_contents": workspace_contents
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

    def build_replan_context(self, original_context: str, previous_plan: dict, executor_response: dict, reviewer_response: dict) -> str:
        '''
            Builds a small amount of context for replanning.

            The goal is not to dump everything into the Planner.
            The goal is only to tell the Planner:
                - what the previous goal was,
                - what tools were used,
                - what final result was produced,
                - why the reviewer rejected it.
        '''
        execution_trace = executor_response.get("execution_trace", [])

        simplified_trace = []

        for step in execution_trace:
            simplified_trace.append({
                "id": step.get("id"),
                "tool": step.get("tool"),
                "args": step.get("args"),
                "status": step.get("status"),
                "result": step.get("result")
            })

        reviewer_issues = reviewer_response.get("issues", [])

        return f"""
                {original_context}

                REPLAN CONTEXT:
                The previous attempt was rejected.

                Previous planner goal:
                {previous_plan.get("goal", "N/A")}

                Previous executed steps:
                {json.dumps(simplified_trace, indent=2)}

                Reviewer issues:
                {json.dumps(reviewer_issues, indent=2)}

                Replanning instruction:
                Create a revised plan for the original current user task only.
                Do not treat the previous plan as the task.
                Do not treat the previous execution as the task.
                Do not copy unrelated file names, paths, tools, or content from the failed attempt.
                Only use this replan context to avoid repeating the same mistake.
                """

    def validate_plan_tool_scope(self, user_task: str, plan_response: dict) -> str | None:
        '''
            Checks that a non-workspace task did not accidentally receive workspace tools.
            This protects the system from Planner mistakes before permission/execution.
        '''
        if self.is_likely_workspace_task(user_task):
            return None

        workspace_tools = {
            "read_file",
            "list_dir",
            "find_file",
            "summarise_txt",
            "create_file",
            "write_file",
            "append_file",
            "delete_file",
            "search_text",
            "replace_text",
            "list_tree",
            "find_file_recursive",
            "create_dir",
            "delete_dir",
            "move_path",
            "copy_path"
        }

        used_workspace_tools = []

        for step in plan_response.get("steps", []):
            tool_name = step.get("tool")

            if tool_name in workspace_tools:
                used_workspace_tools.append(tool_name)

        if used_workspace_tools:
            return (
                f"Planner used workspace tools {used_workspace_tools}, "
                f"but the current task does not ask for file or workspace operations."
            )

        return None

    def replan_after_review_rejection(self, conversation_id: int, user_task: str, plan_response: dict,
                                      executor_response: dict,reviewer_response: dict, approved_actions: set) -> Message:
        '''
            This method is called when reviewer rejects the execution result. It gives the planner the reviewer issues so it can create a better plan.
            IMPORTANT: user_task must stay pure (the original user request). Rejection feedback goes into context only,
            otherwise the workspace heuristic would see workspace-related words in the rejection text and flip incorrectly.
        '''
        issues_text = "; ".join(reviewer_response.get("issues", [])) or "No specific issues provided."

        replan_feedback = (
            f"Previous attempt was rejected because: {issues_text}.\n"
            f"Plan again from scratch for the original user task. "
            f"Use *_step to chain values between steps. "
            f"Do not add steps unrelated to the original task. "
            f"Do not delete or read files the user did not mention."
        )

        context = self.memory.get_relevant_memory(task=user_task, k=5)
        context = f"{context}\n\n{replan_feedback}" if context else replan_feedback

        k_recent_messages = self.memory.get_recent_conversation_messages(conversation_id=conversation_id, k=5)

        planner_input = self.build_planner_input(user_task=user_task, context=context, k_recent_messages=k_recent_messages,
                                                conversation_id=conversation_id, step_index=self.PLANNER_STEP)

        planner_msg = self.try_plan(planner_input=planner_input)
        if planner_msg.status == "failed":
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP, 
                                              review_summary="Replanning failed after reviewer rejected the result.", 
                                              issues=[planner_msg.response.get("error", "Unknown replanning error.")],
                                              execution_response=executor_response, error=planner_msg.response.get("error", "Unknown replanning error."),
                                              details=planner_msg.response
                                              )


        new_plan_response = planner_msg.response
        execution_state = self.executor.initialise_execution_state(plan_response=new_plan_response, context=context, recent_messages=k_recent_messages)
        if approved_actions:
            execution_state["approved_actions"] = approved_actions

        executor_msg = self.run_execution_loop(conversation_id=conversation_id, execution_state=execution_state, start_step_index=self.EXECUTOR_STEP,
                                               user_task=user_task, plan_response=new_plan_response)

        if executor_msg.status == "waiting":
            return executor_msg
        
        if executor_msg.status == "failed":
            execution_state = executor_msg.response.get("execution_state", {})
            rollback_results = self.rollback_execution(executor_response=execution_state)

            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Execution failed during replanning.",
                                              issues=[executor_msg.response.get("error", "Unknown execution error.")],
                                              execution_response=execution_state,
                                              rollback_results=rollback_results,
                                              error=executor_msg.response.get("error", "Unknown execution error."),
                                              details=executor_msg.response)

        return self.continue_workflow(conversation_id=conversation_id, user_task=user_task, plan_response=new_plan_response, executor_msg=executor_msg, 
                                      review_retry_used=True)
    
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
        '''
            Main method which will be called by the client code, as it is going to handle the entire system, i.e. core workflow logic
            Note conv_id will be retrieved by the main entry point where the text sent by user is being handled
        '''
        user_message = Message(conversation_id=conversation_id, step_index=0, sender="user", receiver="coordinator", target_agent=None, message_type="user_message",
                               status="completed", response={"content": user_task}, visibility="external")
        self.memory.store_message(message=user_message)
        

        context = self.memory.get_relevant_memory(task=user_task, k=5)
        k_recent_messages = self.memory.get_recent_conversation_messages(conversation_id=conversation_id, k=5)

        planner_input = self.build_planner_input(user_task = user_task, context = context, k_recent_messages = k_recent_messages, 
                                                 conversation_id = conversation_id, step_index = self.PLANNER_STEP)
        

        planner_msg = self.try_plan(planner_input=planner_input)

        if planner_msg.status == 'failed' and planner_msg.target_agent is None:
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Planning failed.",issues=[planner_msg.response.get("error", "Unknown planning error.")],
                                              error=planner_msg.response.get("error", "Unknown planning error."),
                                              details=planner_msg.response) 
        
        plan_response = self.pin_user_filenames(planner_msg.response, user_task)

        plan_scope_error = self.validate_plan_tool_scope(
            user_task=user_task,
            plan_response=plan_response
        )

        if plan_scope_error:
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Planner produced a plan outside the task scope.",
                                              issues=[plan_scope_error],
                                              details={
                                                "user_task": user_task,
                                                "plan_response": plan_response
                                                }
                                            )

        #sending the respective accepted response to the user so that it knows what it will do and user will have to verify  
        execution_state = self.executor.initialise_execution_state(plan_response=plan_response, context=context, recent_messages=k_recent_messages, user_task=user_task)
        executor_msg = self.run_execution_loop(conversation_id=conversation_id, execution_state=execution_state, start_step_index=self.EXECUTOR_STEP, 
                                       user_task=user_task, plan_response=plan_response)

        if executor_msg.status == "waiting":
            return executor_msg #routing the respective message back to the runner method
        
        if executor_msg.status == "failed":
            execution_state = executor_msg.response.get("execution_state", {})
            rollback_results = self.rollback_execution(executor_response=execution_state)

            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                              review_summary="Execution failed before review.",
                                              issues=[executor_msg.response.get("error", "Unknown execution error.")],
                                              execution_response=execution_state, rollback_results=rollback_results,
                                              error=executor_msg.response.get("error", "Unknown execution error."),
                                              details=executor_msg.response)
        
        return self.continue_workflow(conversation_id=conversation_id, user_task=user_task,plan_response=plan_response, executor_msg=executor_msg,
                                      review_retry_used=False)
    
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

        episode_summary = self.create_epsiodic_summary(user_task=user_task, planner_response=plan_response,
                                                       executor_response=executor_response, reviewer_response= reviewer_response)

        memory_msg = self.memory.store_long_term_memory(user_task=user_task,episode_summary=episode_summary,conversation_id=conversation_id, step_index=self.MEMORY_STEP)
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
            runnable_steps = self.executor.get_runnable_wave(execution_state)

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

            executor_msg = self.executor.run_set_tools(conversation_id=conversation_id, step_index=start_step_index,execution_state=execution_state,runnable_steps=runnable_steps)
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

            resolved_args = self.executor.resolve_step_args_for_permission(
                step=step,
                execution_state=execution_state
            )

            approved_action_signatures.append(
                self.permission_signature(step, resolved_args)
            )

        execution_state["approved_step_ids"].update(step["id"] for step in pending_runnable_steps 
                                                    if self.tool_registry.get(step["tool"], {}).get("requires_permission", False)
                                                    )

        execution_state.setdefault("approved_actions", set()).update(approved_action_signatures)

        #executing the exact runnable wave which was already shown to the user and approved
        executor_msg = self.executor.run_set_tools(conversation_id=conversation_id, step_index=step_index, execution_state=execution_state, 
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
    

    '''
        Episodic summary method - needs checking
    '''
    def create_epsiodic_summary(self, user_task: str, planner_response: dict, executor_response: dict, reviewer_response: dict) -> str:
        """
            Uses the LLM to produce a concise, factual episode summary from the completed workflow. This summary is what the MemoryAgent uses to decide 
            whether something is useful to be considered as long-term semantic memory, and if no outcome is retrieved from llm a fallback summary is coded.
        """
        cleaned_executor_response = self.reviewer.build_review_evidence(executor_response) #using the same funcitonality because it is important here too

        messages = [
            {
                "role": "system",
                "content": (
                    "Given a completed workflow, produce a short factual summary "
                    "describing: what the user asked for, what the plan's goal was, "
                    "what tools were executed and what they produced, and what the "
                    "reviewer concluded. "
                    "Also identify whether the user revealed any stable personal fact, "
                    "preference, project detail, tool preference, learning goal, or reusable task lesson. "
                    "If the user revealed a stable fact or preference, state it clearly in the summary. "
                    "Do not invent facts. "
                    "Be concise (2-4 sentences). Do not add opinions or speculation. "
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User task:\n{user_task}\n\n"
                    f"Planner goal:\n{planner_response.get('goal', 'N/A')}\n\n"
                    f"Execution trace:\n{cleaned_executor_response}\n\n"
                    f"Reviewer summary:\n{reviewer_response.get('review_summary', 'N/A')}"
                ),
            },
        ]
 
        try:
            summary = self.llm_client.invoke_text(messages=messages, stream=False).strip()
            if summary: #if condition to make sure it did not return ""
                return summary 
        except Exception:
            pass  #letting the error slide, as if llm connection does not happen then the pre-coded summary would be sent instead

        goal = planner_response.get("goal", user_task)
        review = reviewer_response.get("review_summary", "completed successfully")
        return f"Goal: {goal}. Outcome: {review}."
    

    def is_likely_workspace_task(self, user_task: str) -> bool:
        """
        It is a heuristic method which helps the model decide whether the task is clearly about workspace/file operations.

        If this returns False, the Planner should only receive direct_response,
        because the user is probably chatting or asking a normal question.
        """
        text = user_task.lower()

        normal_writing_phrases = [
        "write me a letter",
        "write a letter",
        "write me an email",
        "write an email",
        "write me a message",
        "write a message",
        "draft a letter",
        "draft an email",
        ]

        file_keywords = [
            "file",
            "folder",
            "directory",
            "workspace",

            "read file",
            "read the file",
            "read a file",

            "create file",
            "create a file",
            "create new file",

            "write file",
            "write to",
            "save as",
            "save it as",

            "append",
            "append to",

            "overwrite",
            "replace file",
            "replace text",
            "replace in file",

            "delete file",
            "remove file",
            "copy file",
            "copy",
            "duplicate file",
            "duplicate",
            "find file",
            "list files",
            "list folder",
            "search in files",
            "search text",
            "find text",
            "summarise file",
            "summarize file",
            "summarise the file",
            "summarize the file",
            "subdirectory",
            "subdirectories",
            "folder",
            "folders",
            "directory",
            "directories",
            "list tree",
            "project tree",
            "look through",
            "recursive",
            "create folder",
            "create directory",
            "delete folder",
            "delete directory",
            "remove folder",
            "remove directory",
            "move file",
            "move folder",
            "move directory",
            "rename file",
            "rename folder",
            "copy file",
            "copy folder",
            "copy directory",
            "duplicate file",
            "duplicate folder",
        ]

        file_extensions = [
            ".txt",
            ".py",
            ".json",
            ".md",
            ".csv",
            ".docx",
            ".pdf",
        ]

        if any(phrase in text for phrase in normal_writing_phrases):
            if not any(word in text for word in file_keywords):
                return False

        if any(ext in text for ext in file_extensions):
            return True

        if any(keyword in text for keyword in file_keywords):
            return True

        return False
    
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
    
    
    def extract_user_filename(self, user_task: str) -> str | None:
        '''
            Pulls the literal filename out of the user task. Handles:
              - "create me a file called Bye"        -> "Bye"
              - "called work.txt"                    -> "work.txt"
              - "named report.md"                    -> "report.md"
              - "in text mode" / "as txt"            -> appends .txt to bare name
            Returns None when no filename can be confidently extracted.
        '''
        import re

        text = user_task.strip()

        match = re.search(
            r"\b(?:called|named)\s+([A-Za-z0-9_\-/.]+)",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            #Bare "filename.ext" anywhere in the task.
            match = re.search(r"\b([A-Za-z0-9_\-/]+\.[A-Za-z0-9]{1,6})\b", text)

        if not match:
            return None

        candidate = match.group(1).strip(".,'\"")

        #If candidate has no extension, look for a "mode" phrase to attach one.
        if "." not in candidate:
            lower_text = text.lower()
            for phrase, ext in self.MODE_EXTENSIONS.items():
                if phrase in lower_text:
                    candidate = f"{candidate}.{ext}"
                    break

        return candidate

    def pin_user_filenames(self, plan_response: dict, user_task: str) -> dict:
        '''
            Deterministically overwrites planner-provided paths with the literal
            filename extracted from the user task. Stops the planner from
            substituting names like "file_to_create" or "empty_file.txt".
        '''
        pinned_name = self.extract_user_filename(user_task)
        if not pinned_name:
            return plan_response

        for step in plan_response.get("steps", []):
            if step.get("tool") not in self.PATH_PINNED_TOOLS:
                continue

            args = step.get("args", {})
            current = args.get("path", "")

            #Strip markdown-link syntax the planner sometimes wraps filenames in.
            if isinstance(current, str):
                import re
                md = re.match(r"^\s*\[([^\]]+)\]\([^)]*\)\s*$", current)
                if md:
                    current = md.group(1)

            if current != pinned_name:
                args["path"] = pinned_name
                step["args"] = args

        return plan_response