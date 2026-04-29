from src.agents.planning.PlannerAgent import PlannerAgent
from src.agents.planning.PlanCompiler import PlanCompiler
from src.agents.execution.ExecutorAgent import ExecutorAgent
from src.core.workflow.ExecutionVerifier import ExecutionVerifier
from src.agents.memory.MemoryAgent import MemoryAgent
from src.agents.reviewing.ReviewerAgent import ReviewerAgent
from src.agents.routing.RouteAgent import RouteAgent
from src.agents.routing.WorkflowPolicy import WorkflowPolicyRegistry
from src.agents.routing.MemoryRoutingPolicy import MemoryRoutingPolicy
from src.core.message import Message
from src.llm.OllamaClient import OllamaClient
import json
import re

'''
    This is the agent which manages workflow, and configures the respective agents i.e. it acts like the runner method,
    and it tries to enforce certain heuristic methods which try to help the model be restricted in the way it can think (try to plan out something deterministicly
    this is because of hallucinations)
'''
class Coordinator():
    MAX_WORKFLOW_ITERATIONS = 3
    MAX_EXECUTION_LOOP_ITERATIONS = 25
    MAX_REPEATED_EXECUTION_STATES = 2

    ROUTER_STEP = 1
    PLANNER_STEP = 2
    EXECUTOR_STEP = 3
    REVIEWER_STEP = 4
    MEMORY_STEP = 5
    FINAL_STEP = 6

    """
        Construction helpers shared by every workflow path.
    """

    def __init__(self, planner: PlannerAgent, executor: ExecutorAgent, reviewer: ReviewerAgent,
                 memory: MemoryAgent, router: RouteAgent, planner_tool_descriptions: list[dict], tool_registry: dict, llm_client: OllamaClient,
                 execution_verifier: ExecutionVerifier = None):
        self.planner = planner
        self.planner_tool_descriptions = planner_tool_descriptions
        self.executor = executor
        self.tool_registry = tool_registry
        self.reviewer = reviewer
        self.memory = memory
        self.router = router
        self.llm_client = llm_client
        self.execution_verifier = execution_verifier

    """
        Routing, scope, and planner input builders that decide what context and tools each task may use.
    """

    def run_router(self, conversation_id: int, user_task: str) -> dict:
        '''
            Router only classifies the task.
            Coordinator converts that classification into a scoped workflow route.
        '''

        route_msg = self.router.run(conversation_id=conversation_id, step_index=self.ROUTER_STEP, user_task=user_task)
        self.memory.store_message(route_msg)

        if route_msg.status != "completed":
            return WorkflowPolicyRegistry.build_route({
                "task_type": "direct_response",
                "confidence": 0.0,
                "routing_reason": "Router failed, so Coordinator used safe fallback."
            })

        return WorkflowPolicyRegistry.build_route(route_msg.response)

    def validate_route(self, route: dict) -> dict:
        """
            Kept for compatibility, but route validation is now just rebuilding
            from task_type using infrastructure-owned policy.
        """

        return WorkflowPolicyRegistry.build_route(route)

    def select_tools_for_route(self, route: dict) -> list[dict]:
        allowed_tools = set(route.get("allowed_tools", []))

        if not allowed_tools:
            allowed_tools = WorkflowPolicyRegistry.DIRECT_RESPONSE_TOOLS

        return [tool for tool in self.planner_tool_descriptions if tool["name"] in allowed_tools]

    def build_scoped_context(self, user_task: str, route: dict) -> str:
        decision = MemoryRoutingPolicy.decide(user_task=user_task, route=route)
        route["memory_routing"] = {
            "use_short_term": decision.use_short_term,
            "use_long_term": decision.use_long_term,
            "allowed_memory_types": list(decision.allowed_memory_types),
            "long_term_mode": decision.long_term_mode,
            "long_term_k": decision.long_term_k,
            "short_term_k": decision.short_term_k,
            "reason": decision.reason,
        }

        if not decision.use_long_term:
            return ""

        # Memory routing is deterministic: task type and simple keyword signals
        # decide whether durable memory is visible to the planner/responder.
        if hasattr(self.memory, "get_memory_by_mode"):
            context = self.memory.get_memory_by_mode(
                task=user_task,
                mode=decision.long_term_mode,
                k=decision.long_term_k,
                allowed_memory_types=decision.allowed_memory_types
            )
        elif hasattr(self.memory, "get_relevant_memory"):
            # Compatibility for simple test doubles. Production MemoryAgent
            # uses get_memory_by_mode so all routing decisions stay centralized.
            context = self.memory.get_relevant_memory(task=user_task, k=decision.long_term_k)
        else:
            context = ""

        if context is None:
            return ""

        if isinstance(context, list):
            return "\n".join(str(item) for item in context)

        return str(context)

    def build_scoped_recent_messages(self, conversation_id: int, route: dict) -> list[dict]:
        decision_data = route.get("memory_routing")

        if decision_data is None:
            decision = MemoryRoutingPolicy.decide(user_task="", route=route)
            use_short_term = decision.use_short_term
            short_term_k = decision.short_term_k
        else:
            use_short_term = decision_data.get("use_short_term", False)
            short_term_k = decision_data.get("short_term_k", 0)

        if not use_short_term:
            return []
        
        return self.memory.get_recent_conversation_messages(conversation_id=conversation_id, k=short_term_k or 5)

    def build_scoped_workspace_contents(self, route: dict):
        if not route["use_workspace"]:
            return []
        
        list_tree = self.tool_registry.get("list_tree", {}).get("func")
        if list_tree:
            return list_tree(path=".")
        
        return self.tool_registry["list_dir"]["func"](path=".")

    def build_planner_input(self, user_task: str, route: dict, conversation_id: int, step_index: int, replan_feedback: str = "") -> dict:
        scoped_context = self.build_scoped_context(user_task, route)
        if replan_feedback:
            scoped_context = f"{scoped_context}\n\nREPLAN CONTEXT\n{replan_feedback}" if scoped_context else f"REPLAN CONTEXT:\n{replan_feedback}"

        scoped_recent_messages = self.build_scoped_recent_messages(conversation_id, route)
        selected_tools = self.select_tools_for_route(route)
        scoped_workspace_contents = self.build_scoped_workspace_contents(route)
        requested_paths, allowed_parent_dirs = self.extract_requested_path_constraints(user_task=user_task, route=route, selected_tools=selected_tools)

        return {
            "task": user_task,
            "context": scoped_context,
            "k_recent_messages": scoped_recent_messages,
            "tools": selected_tools,
            "workspace_contents": scoped_workspace_contents,
            "requested_paths": requested_paths,
            "allowed_parent_dirs": allowed_parent_dirs,
            "route": route,
            "conversation_id": conversation_id,
            "step_index": step_index,
        }

    def extract_requested_path_constraints(self, user_task: str, route: dict, selected_tools: list[dict]) -> tuple[list[str], list[str]]:
        if route.get("task_type") != "workspace_mutation":
            return [], []

        path_helper = PlanCompiler(
            available_tools=selected_tools,
            workspace_config=None,
            route=route,
            user_task=user_task
        )

        requested_paths = path_helper.extract_explicit_file_paths(user_task)
        allowed_paths = path_helper.allowed_paths_for_requested_files(requested_paths)
        allowed_parent_dirs = sorted(path for path in allowed_paths if path not in set(requested_paths))

        return requested_paths, allowed_parent_dirs

    """
        Planning retry behaviour that turns compiler failures into focused replan feedback.
    """

    def build_path_retry_feedback(self, planner_input: dict, error: str) -> str:
        requested_paths = planner_input.get("requested_paths") or []

        if not requested_paths or "planned mutation path" not in error:
            return ""

        allowed_parent_dirs = planner_input.get("allowed_parent_dirs") or []
        used_path = ""
        match = re.search(r"planned mutation path '([^']+)'", error)
        if match:
            used_path = match.group(1)

        allowed_mutation_paths = sorted(set(allowed_parent_dirs + requested_paths))

        if len(requested_paths) == 1:
            requested_path = requested_paths[0]
            allowed_text = " and ".join(allowed_mutation_paths)
            feedback = (
                f"The user explicitly requested {requested_path}. "
                f"Your plan used {used_path or 'a different path'}. "
                f"Regenerate the plan using exactly {requested_path}. "
                f"The only allowed mutation paths are {allowed_text}. "
                f"Do not invent hello_world.py, code.py, main.py, or any other filename."
            )

            if allowed_parent_dirs:
                feedback += f" If a parent directory is needed, create only '{allowed_parent_dirs[0]}'."

            return feedback

        requested_text = ", ".join(f"'{path}'" for path in requested_paths)
        feedback = (
            f"The user explicitly requested these paths: {requested_text}. "
            f"Every file mutation must use only those exact requested paths. "
            f"Do not invent hello_world.py, code.py, main.py, or any other filename."
        )

        if allowed_parent_dirs:
            parent_text = ", ".join(f"'{path}'" for path in allowed_parent_dirs)
            feedback += f" If parent directories are needed, create only: {parent_text}."

        return feedback

    def build_step_reference_retry_feedback(self, error: str) -> str:
        if "fake step reference" not in error:
            return ""

        if "'content'" in error and "content_step" in error:
            return (
                "You used {'content': 'content_step'}, which is invalid. "
                "Use {'content_step': 1} instead when content comes from step 1."
            )

        return (
            "Do not put *_step markers as string values inside normal args. "
            "Use {'content_step': 1}, {'path_step': 1}, or {'text_step': 1} with an integer step id."
        )

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

                path_retry_feedback = self.build_path_retry_feedback(planner_input=retry_planner_input, error=error)
                step_reference_retry_feedback = self.build_step_reference_retry_feedback(error=error)
                
                #Giving the planner retry some feedback about why the previous plan failed compilation
                retry_planner_input["context"] += f"""

                REPLAN CONTEXT:
                The previous plan failed compilation.

                Compiler error:
                {error}

                Path constraints:
                {path_retry_feedback}

                Step reference constraints:
                {step_reference_retry_feedback}

                Fix the plan structure only.

                Rules:
                - Keep the original CURRENT TASK unchanged.
                - Use only available tools.
                - Do not invent files, paths, or content.
                - Use direct args when the value is known from the current task.
                - Use *_step only when the value comes from an earlier step.
                - *_step must reference an earlier integer step id.
                - Do not add unrelated steps.
                """

            return planner_msg  #failed after retries

    """
        Execution loop behaviour, including loop guards, permission requests, permission continuations, and rollback support.
    """

    def run_execution_loop(self, conversation_id: int, execution_state: dict, start_step_index: int, user_task: str, plan_response: dict) -> Message:
        '''
            Interleaving with the executor agent in order to tell it when to run, since certain tasks would need permission before it does so.
        '''

        #Initialising it to this respective message if while loop does not re-set execution_msg to the corresponding message
        executor_msg = Message(conversation_id=conversation_id, step_index=start_step_index, sender="coordinator", receiver="user", target_agent=None, 
                               message_type="workflow_result",status="failed", 
                               response={"message": "Planner produced no executable steps."}, visibility="external")
        
        loop_iterations = 0
        seen_states = {}

        while not self.executor.is_execution_complete(execution_state):
            loop_iterations += 1
            execution_state["execution_loop_iterations"] = loop_iterations

            if loop_iterations > self.MAX_EXECUTION_LOOP_ITERATIONS:
                error = f"Execution loop guard triggered after {loop_iterations} iterations."
                execution_state["loop_guard"] = {
                    "reason": "max_execution_loop_iterations_exceeded",
                    "max_execution_loop_iterations": self.MAX_EXECUTION_LOOP_ITERATIONS,
                    "iterations": loop_iterations
                }
                return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                                  review_summary="Execution stopped by loop guard.",
                                                  issues=[error], execution_response=execution_state,
                                                  error=error, details=execution_state["loop_guard"])

            state_signature = self.execution_state_signature(execution_state=execution_state)
            seen_states[state_signature] = seen_states.get(state_signature, 0) + 1

            if seen_states[state_signature] > self.MAX_REPEATED_EXECUTION_STATES:
                error = "Execution repetition guard triggered: the same execution state repeated."
                execution_state["loop_guard"] = {
                    "reason": "repeated_execution_state",
                    "signature": state_signature,
                    "repeat_count": seen_states[state_signature],
                    "max_repeated_execution_states": self.MAX_REPEATED_EXECUTION_STATES
                }
                return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                                  review_summary="Execution stopped by repetition guard.",
                                                  issues=[error], execution_response=execution_state,
                                                  error=error, details=execution_state["loop_guard"])

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

            executor_msg = self.executor.run_steps(conversation_id=conversation_id, step_index=start_step_index,execution_state=execution_state,
                                                   runnable_steps=runnable_steps)
            self.memory.store_message(executor_msg)

            if executor_msg.status == "failed":
                return executor_msg
            
            if executor_msg.message_type == "execution_wave_result":
                execution_state = executor_msg.response
                continue

        return executor_msg

    def execution_state_signature(self, execution_state: dict) -> str:
        """
            Small deterministic signature used by repetition guards. It ignores
            bulky result values and focuses on whether the loop is making
            progress through remaining steps and dependency status.
        """
        remaining_ids = tuple(step.get("id") for step in execution_state.get("remaining_steps", []))
        statuses = tuple(sorted((str(key), value) for key, value in execution_state.get("step_status", {}).items()))
        approved_ids = tuple(sorted(execution_state.get("approved_step_ids", set())))
        approved_actions = tuple(sorted(execution_state.get("approved_actions", set())))

        return json.dumps({
            "remaining_ids": remaining_ids,
            "statuses": statuses,
            "approved_ids": approved_ids,
            "approved_actions": approved_actions,
        }, sort_keys=True)

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
                "resolved_args": self.executor.resolve_step_args_for_permission(step=step,execution_state=execution_state),
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
                              executor_msg=executor_msg, workflow_iteration=execution_state.get("workflow_iteration", 1))
        
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
                                          executor_msg=next_executor_msg, workflow_iteration=updated_execution_state.get("workflow_iteration", 1))
        
        #Fall back return statement if something unexpected breaks inside the workflow
        return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                          review_summary="Unexpected breakage in the workflow.",
                                          issues=["The executor returned an unexpected message type."],
                                          error="Unexpected breakage in the workflow.",
                                          details=executor_msg.response)

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

    def cleanup_rejected_execution_state(self, executor_response: dict) -> dict:
        """
            Remove execution state that belongs only to a rejected attempt.

            Replans must start from a fresh execution state. In particular,
            permissions approved for rejected mutation paths must never carry into
            a later corrected plan.
        """
        cleaned_response = dict(executor_response or {})

        cleaned_response["approved_actions"] = []
        cleaned_response["approved_step_ids"] = []
        cleaned_response["remaining_steps"] = []
        cleaned_response["step_results"] = {}
        cleaned_response["step_status"] = {}
        cleaned_response["execution_trace"] = []

        return cleaned_response

    """
        Review, verifier, memory, and final response behaviour after execution has produced a result.
    """

    def continue_workflow(self, conversation_id: int, user_task: str, plan_response: dict, executor_msg: Message,
                          review_retry_used: bool = None, workflow_iteration: int = 1) -> Message:
        '''
            This method is handling the respective logic which happens after permission is accepted (if needed), so that the user is kept in the loop while
            also not having redundant lines of code in the coordinator
        '''

        # Backwards-compatible handling for older tests/callers that pass
        # review_retry_used=True. Internally, iteration number is now the source
        # of truth for bounded workflow retries.
        if review_retry_used is True:
            workflow_iteration = max(workflow_iteration, self.MAX_WORKFLOW_ITERATIONS)

        executor_response = executor_msg.response #will only contain messages with .status == "completed"
        executor_response["workflow_iteration"] = workflow_iteration

        #Reviewer needs to see the actual workspace state to verify set-based claims
        #like "delete all files except X" — the trace alone is insufficient.
        list_tree = self.tool_registry.get("list_tree", {}).get("func")
        workspace_after = list_tree(path=".") if list_tree else []
        workspace_before = executor_response.get("workspace_before", [])

        if self.execution_verifier is not None:
            verifier_response = self.execution_verifier.verify(executor_response=executor_response)

            if not verifier_response.get("accepted", False):
                rollback_results = self.rollback_execution(executor_response=executor_response)

                self.memory.store_message(message=Message(conversation_id=conversation_id, step_index=self.REVIEWER_STEP,
                                                        sender="verifier", receiver="coordinator", target_agent=None,
                                                        message_type="verification_result", status="completed",
                                                        response=verifier_response, visibility="internal"))

                self.memory.store_message(message=Message(conversation_id=conversation_id, step_index=self.MEMORY_STEP,
                                                        sender="memory", receiver="coordinator", target_agent=None,
                                                        message_type="memory_store", status="completed",
                                                        response={"stored": False, "reason": "Execution verifier did not accept workflow"},
                                                        visibility="internal"))

                issues = verifier_response.get("issues", [])
                if issues and workflow_iteration < self.MAX_WORKFLOW_ITERATIONS:
                    cleaned_executor_response = self.cleanup_rejected_execution_state(executor_response)
                    return self.replan_after_review_rejection(conversation_id=conversation_id, user_task=user_task,
                                                            plan_response=plan_response, executor_response=cleaned_executor_response,
                                                            reviewer_response=verifier_response,
                                                            workflow_iteration=workflow_iteration + 1)

                return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                                review_summary=verifier_response.get("review_summary", "Execution verifier rejected execution."),
                                                issues=issues or ["Execution verifier rejected execution without concrete issues."],
                                                execution_response=executor_response, rollback_results=rollback_results,
                                                details={
                                                    **verifier_response,
                                                    "workflow_iteration": workflow_iteration,
                                                    "max_workflow_iterations": self.MAX_WORKFLOW_ITERATIONS,
                                                    "retry_limit_reached": workflow_iteration >= self.MAX_WORKFLOW_ITERATIONS
                                                })

        reviewer_msg = self.reviewer.run(conversation_id=conversation_id, step_index=self.REVIEWER_STEP, user_task=user_task, 
                                         execution_response=executor_response, workspace_before=workspace_before, workspace_after=workspace_after,
                                         route=plan_response.get("route", {}))
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

            if issues and workflow_iteration < self.MAX_WORKFLOW_ITERATIONS:
                cleaned_executor_response = self.cleanup_rejected_execution_state(executor_response)
                return self.replan_after_review_rejection(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response,
                                                        executor_response=cleaned_executor_response, reviewer_response=reviewer_response,
                                                        workflow_iteration=workflow_iteration + 1)

            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP, 
                                            review_summary=reviewer_response.get("review_summary", "Reviewer rejected execution."),
                                            issues=issues or ["Reviewer rejected execution without concrete issues."],
                                            execution_response=executor_response, rollback_results=rollback_results,
                                            details={
                                                **reviewer_response,
                                                "workflow_iteration": workflow_iteration,
                                                "max_workflow_iterations": self.MAX_WORKFLOW_ITERATIONS,
                                                "retry_limit_reached": workflow_iteration >= self.MAX_WORKFLOW_ITERATIONS
                                            })

        memory_msg = self.memory.store_long_term_memory(user_task=user_task, conversation_id=conversation_id, step_index=self.MEMORY_STEP)
        self.memory.store_message(message=memory_msg)

        return self.build_success_message(conversation_id=conversation_id, step_index=self.FINAL_STEP,
                                          review_summary=reviewer_response.get("review_summary", "Execution accepted."), 
                                          execution_response=executor_response)

    def replan_after_review_rejection(self, conversation_id: int, user_task: str, plan_response: dict,
                                      executor_response: dict, reviewer_response: dict, workflow_iteration: int) -> Message:
        '''
            user_task stays pure. Rejection feedback goes into context only.
            Route is reused so replanning stays inside the same scoped environment.
        '''
        issues = reviewer_response.get("issues", [])
        issues_text = "; ".join(issues) or "No specific issues provided."
        previous_trace = executor_response.get("execution_trace", [])
        current_workspace = self.tool_registry["list_tree"]["func"](path=".")

        replan_feedback = (
            f"Previous attempt was rejected. "
            f"Previous workflow iteration {workflow_iteration - 1} was rejected because: {issues_text}.\n"
            f"Previous plan:\n{json.dumps(plan_response, indent=2, default=str)}\n"
            f"Previous execution trace:\n{json.dumps(previous_trace, indent=2, default=str)}\n"
            f"Current workspace state:\n{json.dumps(current_workspace, indent=2, default=str)}\n"
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
        new_plan_response["workflow_iteration"] = workflow_iteration

        execution_state = self.executor.initialise_execution_state(plan_response=new_plan_response, context=planner_input["context"], 
                                                                   recent_messages=planner_input["k_recent_messages"], user_task=user_task, route=route)

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
                                      plan_response=new_plan_response, executor_msg=executor_msg,
                                      workflow_iteration=workflow_iteration)


    '''
        Messages sent to the runner file in order for the response field to be sent to the user in order for it to know what happened with its task
    '''

    def build_success_message(self, conversation_id: int, step_index: int, review_summary: str, execution_response: dict) -> Message:
        trace = execution_response.get("execution_trace", [])
        trace = sorted(trace, key=lambda step: (step.get("workflow_iteration", 1), step["id"]))

        direct_outputs = [step.get("result") for step in trace if step.get("tool") == "direct_response" and step.get("status") == "completed" and step.get("result")]

        display_tools = {"read_file", "list_dir", "find_file", "summarise_txt", "create_file", "write_file", "append_file", "delete_file", 
                         "search_text", "replace_text","create_dir","delete_dir", "move_path", "copy_path","find_file_recursive", "list_tree"}

        display_outputs = [step.get("result") for step in trace if step.get("tool") in display_tools and step.get("status") == "completed" 
                           and step.get("result") is not None]
        
        if direct_outputs:
            display_result = direct_outputs[-1]
        elif display_outputs:
            display_result = "\n".join(str(output) for output in display_outputs)
        else:
            display_result = None

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
        trace = sorted(trace, key=lambda step: (step.get("workflow_iteration", 1), step["id"])) if trace else []

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

    """
        Top-level workflow entry point that records the user request and starts routing, planning, execution, and review.
    """

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
        plan_response["workflow_iteration"] = 1

        execution_state = self.executor.initialise_execution_state(plan_response=plan_response, context=planner_input["context"], 
                                                                   recent_messages=planner_input["k_recent_messages"], user_task=user_task, route=route)
    
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
                                      plan_response=plan_response, executor_msg=executor_msg,
                                      workflow_iteration=1)
