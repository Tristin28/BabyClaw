from src.Agents.Planning.PlannerAgent import PlannerAgent
from src.Agents.ExecutorAgent import ExecutorAgent
from src.Agents.MemoryAgent import MemoryAgent
from src.Agents.ReviewerAgent import ReviewerAgent
from src.message import Message
from src.OllamaClient import OllamaClient
from src.tools.file_tools import list_dir


'''
    This is the agent which manages workflow, and configures the respective agents i.e. it acts like the runner method
'''
class Coordinator():
    PLANNER_STEP = 1
    EXECUTOR_STEP = 2
    REVIEWER_STEP = 3
    MEMORY_STEP = 4
    FINAL_STEP = 5
     
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

        available_tools = self.planner_tool_descriptions
        if not self.is_likely_workspace_task(user_task):
            available_tools = [tool for tool in self.planner_tool_descriptions if tool["name"] == "direct_response"]

        return {"task": user_task, "context": context, "k_recent_messages": k_recent_messages, "tools": available_tools, 
                "conversation_id": conversation_id, "step_index": step_index, "workspace_contents": workspace_contents}

    def try_plan(self,planner_input: dict, max_attempts: int = 3) -> Message:
            '''
                Method which handles the raised exceptions for when the planner is running so that it manages the respective cases by replaning or 
                else tells coordinator that it can't handle this respective task.
            '''
            for _ in range(max_attempts):
                planner_msg = self.planner.run(planner_input=planner_input)
                self.memory.store_message(planner_msg)

                if planner_msg.status == "completed":
                    return planner_msg
                
                error = planner_msg.response.get("error", "")
                if "Unknown tool" in error:
                    return planner_msg   #stop immediately
                
                planner_input["context"] += f"""
                    Previous planning attempt failed.

                    Compiler error:
                    {error}

                    Fix the plan and try again.

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
    
    def replan_after_review_rejection(self, conversation_id: int, user_task: str, plan_response: dict, executor_response: dict,reviewer_response: dict) -> Message:
        '''
            This method is called when reviewer rejects the execution result. It gives the planner the reviewer issues so it can create a better plan.
        '''
        revised_task = {
            "original_task": user_task,
            "previous_plan": plan_response,
            "previous_execution": executor_response,
            "reviewer_issues": reviewer_response.get("issues", []),
            "instruction": "Create a revised plan that fixes the reviewer issues. Use only the available tools."
        }

        context = self.memory.get_relevant_memory(task=user_task, k=5)
        k_recent_messages = self.memory.get_recent_messages(conversation_id=conversation_id, k=5)

        planner_input = self.build_planner_input(user_task=str(revised_task), context=context, k_recent_messages=k_recent_messages,
                                                conversation_id=conversation_id, step_index=self.PLANNER_STEP)

        planner_msg = self.try_plan(planner_input=planner_input)
        if planner_msg.status == "failed":
            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP, 
                                              review_summary="Replanning failed after reviewer rejected the result.", 
                                              issues=[planner_msg.response.get("error", "Unknown replanning error.")]
                                              )


        new_plan_response = planner_msg.response
        execution_state = self.executor.initialise_execution_state(plan_response=plan_response, context=context, recent_messages=k_recent_messages)

        executor_msg = self.run_execution_loop(conversation_id=conversation_id, execution_state=execution_state, start_step_index=self.EXECUTOR_STEP,
                                               user_task=user_task, plan_response=new_plan_response)

        if executor_msg.status == "waiting" or executor_msg.status == "failed":
            return executor_msg

        return self.continue_workflow(conversation_id=conversation_id, user_task=user_task, plan_response=new_plan_response,
                                    executor_msg=executor_msg, step_index=self.REVIEWER_STEP, review_retry_used=True)
    

    '''
        Messages sent to the runner file in order for the response field to be sent to the user in order for it to know what happened with its task
    '''
    def build_success_message(self, conversation_id: int, step_index: int, review_summary: str, execution_response: dict) -> Message:
        trace = execution_response.get("execution_trace", [])
        trace = sorted(trace, key=lambda step: step["id"])

        message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, 
                           message_type="workflow_result", status="completed",
                           response={"message": "Task completed successfully.", "review_summary": review_summary, "execution_trace": trace}, 
                           visibility="external")
        
        self.memory.store_message(message)

        return message

    def build_failure_message(self, conversation_id: int, step_index: int, review_summary: str, issues: list[str]) -> Message:
        message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, message_type="workflow_result",
                           status="failed", response={"message": "Task could not be completed successfully.", "review_summary": review_summary, "issues": issues}
                           ,visibility="external")
        
        self.memory.store_message(message)

        return message


    def start_workflow(self, conversation_id: int, user_task: str):
        '''
            Main method which will be called by the client code, as it is going to handle the entire system, i.e. core workflow logic
            Note conv_id will be retrieved by the main entry point where the text sent by user is being handled
        '''
        
        context = self.memory.get_relevant_memory(task=user_task, k=5)
        k_recent_messages = self.memory.get_recent_messages(conversation_id=conversation_id, k=5)

        planner_input = self.build_planner_input(user_task = user_task, context = context, k_recent_messages = k_recent_messages, 
                                                 conversation_id = conversation_id, step_index = self.PLANNER_STEP)
        

        planner_msg = self.try_plan(planner_input=planner_input)

        if planner_msg.status == 'failed' and planner_msg.target_agent is None:
            return self.build_failure_message(conversation_id=conversation_id,step_index=self.FINAL_STEP,review_summary=planner_msg.response.get("error"), issues=[]) 
        
        plan_response = planner_msg.response

        #sending the respective accepted response to the user so that it knows what it will do and user will have to verify  
        execution_state = self.executor.initialise_execution_state(plan_response=plan_response, context=context, recent_messages=k_recent_messages)
        executor_msg = self.run_execution_loop(conversation_id=conversation_id, execution_state=execution_state, start_step_index=self.EXECUTOR_STEP, 
                                       user_task=user_task, plan_response=plan_response)

        if executor_msg.status=="failed" or executor_msg.status == "waiting":
            return executor_msg #routing the respective message back to the runner method
        
        return self.continue_workflow(conversation_id=conversation_id, user_task=user_task,plan_response=plan_response, executor_msg=executor_msg, step_index=self.REVIEWER_STEP,
                                      review_retry_used=False)
    
    def continue_workflow(self, conversation_id: int, user_task: str, plan_response: dict, executor_msg: Message, 
                          step_index: int, review_retry_used: bool) -> Message:
        '''
            This method is handling the respective logic which happens after permission is accepted (if needed), so that the user is kept in the loop while
            also not having redundant lines of code in the coordinator
        '''

        executor_response = executor_msg.response #will only contain messages with .status == "completed"
        
        reviewer_msg = self.reviewer.run(conversation_id=conversation_id,step_index=self.REVIEWER_STEP, user_task = user_task, execution_trace = executor_response)
        self.memory.store_message(message=reviewer_msg)

        reviewer_response = reviewer_msg.response
        accepted = reviewer_msg.response.get("accepted", False)

        if not accepted:
            self.memory.store_message(message=Message(conversation_id=reviewer_msg.conversation_id, step_index=reviewer_msg.step_index, 
                                                    sender="memory", receiver="coordinator", target_agent=None, message_type="memory_store", 
                                                    status="completed", response={"stored": False, "reason": "Reviewer did not accept workflow"},
                                                    visibility="internal"))

            if not review_retry_used:
                return self.replan_after_review_rejection(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response,
                                                        executor_response=executor_response, reviewer_response=reviewer_response)

            return self.build_failure_message(conversation_id=conversation_id, step_index=self.FINAL_STEP, 
                                            review_summary=reviewer_response.get("review_summary", "Reviewer rejected execution."),
                                            issues=reviewer_response.get("issues", []))
        
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
                message = Message(conversation_id=conversation_id, step_index=start_step_index, sender="coordinator", receiver="user", target_agent=None, 
                               message_type="execution_failed",status="failed", 
                               response={"message": "Execution blocked: no runnable steps found."}, visibility="external")
                
                self.memory.store_message(message)

                return message

            permission_steps = self.get_permission_required_steps(runnable_steps,execution_state=execution_state)

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
        
        execution_state["approved_step_ids"].update(step["id"] for step in pending_runnable_steps 
                                                    if self.tool_registry.get(step["tool"], {}).get("requires_permission", False))

        #executing the exact runnable wave which was already shown to the user and approved
        executor_msg = self.executor.run_set_tools(conversation_id=conversation_id, step_index=step_index, execution_state=execution_state, 
                                                   runnable_steps=pending_runnable_steps)
        self.memory.store_message(message=executor_msg)

        #if execution finished right after the approved wave then continue to reviewer/memory/final response
        if executor_msg.message_type == "execution_result":
            return self.continue_workflow(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response, 
                                          executor_msg=executor_msg, step_index=self.REVIEWER_STEP, review_retry_used=False)
        
        #Handling the case that if when the set of permitted runnable tools fail when executing it stops
        if executor_msg.status == "failed":
            return executor_msg

        #if the approved wave completed but there are still remaining steps then continue the execution loop normally
        if executor_msg.message_type == "execution_wave_result":
            updated_execution_state = executor_msg.response

            next_executor_msg = self.run_execution_loop(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response, 
                                                        execution_state=updated_execution_state, start_step_index=step_index)

            if next_executor_msg.status == "waiting" or next_executor_msg.status=="failed":
                return next_executor_msg #will re-rout the respective permission request back to user again

            return self.continue_workflow(conversation_id=conversation_id, user_task=user_task, plan_response=plan_response, 
                                          executor_msg=next_executor_msg, step_index=self.REVIEWER_STEP, review_retry_used=False)
        
        #Fall back return statement if something unexpected breaks inside the workflow
        message = Message(conversation_id=conversation_id, step_index=step_index, sender="coordinator", receiver="user", target_agent=None, message_type="workflow_result",
                       status="failed",response={"message": "Unexpected breakage in the workflow", "details": executor_msg.response}, visibility="external")
        
        self.memory.store_message(message)
        return message
    
    '''
        Helper functions for the respective permission steps
    '''
    def get_permission_required_steps(self, runnable_steps: list[dict], execution_state: dict) -> list[dict]:
            permission_steps = []

            approved_ids = execution_state.get("approved_step_ids", set())
            for step in runnable_steps:
                tool_name = step["tool"]
                tool_def = self.tool_registry.get(tool_name, {})

                if tool_def.get("requires_permission", False) and step["id"] not in approved_ids:
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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarisation agent. "
                    "Given a completed workflow, produce a short factual summary "
                    "describing: what the user asked for, what the plan's goal was, "
                    "what tools were executed and what they produced, and what the "
                    "reviewer concluded. "
                    "Be concise (2-4 sentences). Do not add opinions or speculation. "
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User task:\n{user_task}\n\n"
                    f"Planner goal:\n{planner_response.get('goal', 'N/A')}\n\n"
                    f"Execution trace:\n{executor_response}\n\n"
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
        Decides whether the task is clearly about workspace/file operations.

        If this returns False, the Planner should only receive direct_response,
        because the user is probably chatting or asking a normal question.
        """
        text = user_task.lower()

        file_keywords = [
            "file",
            "folder",
            "directory",
            "workspace",
            "read",
            "create",
            "write to",
            "save",
            "append",
            "overwrite",
            "replace",
            "delete",
            "list",
            "find file",
            "summarise file",
            "summarize file",
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

        if any(ext in text for ext in file_extensions):
            return True

        if any(keyword in text for keyword in file_keywords):
            return True

        return False