from src.core.workflow.Coordinator import Coordinator
from src.agents.execution.ExecutorAgent import ExecutorAgent
from src.agents.routing.MemoryRoutingPolicy import MemoryRoutingPolicy
from src.core.message import Message


class NoopMemory:
    def __init__(self):
        self.messages = []

    def store_message(self, message):
        self.messages.append(message)
        return message

    def get_memory_by_mode(self, task, mode, k=5, allowed_memory_types=None):
        return ""


class NoopRouter:
    def run(self, conversation_id, step_index, user_task):
        return Message(conversation_id, step_index, "router", "coordinator", None, "route", "completed", {
            "task_type": "direct_response",
            "confidence": 1.0,
            "routing_reason": "test",
        }, "internal")


class NoopPlanner:
    def run(self, planner_input):
        return Message(planner_input["conversation_id"], planner_input["step_index"], "planner", "coordinator",
                       "executor", "plan", "completed", {
                           "goal": "noop",
                           "steps": [{"id": 1, "tool": "direct_response", "args": {"prompt": "hello"}, "depends_on": []}],
                           "planning_rationale": "test",
                       }, "internal")


class NoopReviewer:
    def run(self, conversation_id, step_index, user_task, execution_response, workspace_before=None, workspace_after=None, route=None):
        return Message(conversation_id, step_index, "reviewer", "coordinator", None, "review_result", "completed", {
            "accepted": True,
            "review_summary": "ok",
            "issues": [],
        }, "internal")


def make_minimal_coordinator(executor):
    return Coordinator(
        planner=NoopPlanner(),
        executor=executor,
        reviewer=NoopReviewer(),
        memory=NoopMemory(),
        router=NoopRouter(),
        planner_tool_descriptions=[{
            "name": "direct_response",
            "description": "test",
            "args_schema": {"prompt": {"type": "string", "required": True, "step_chainable": True}},
            "returns": {"type": "string"},
        }],
        tool_registry={
            "list_tree": {
                "func": lambda path=".": [],
                "input_map": {"path": "path"},
                "requires_permission": False,
            },
            "direct_response": {
                "func": lambda prompt: "hello",
                "input_map": {"prompt": "prompt"},
                "requires_permission": False,
            },
        },
        llm_client=None,
    )


def test_memory_routing_keeps_simple_direct_response_memory_free():
    decision = MemoryRoutingPolicy.decide(
        user_task="Explain recursion",
        route={"task_type": "direct_response"}
    )

    assert decision.use_short_term is False
    assert decision.use_long_term is False
    assert decision.long_term_mode == "none"


def test_memory_routing_strengthens_memory_question():
    decision = MemoryRoutingPolicy.decide(
        user_task="What do you remember about me?",
        route={"task_type": "memory_question"}
    )

    assert decision.use_long_term is True
    assert decision.long_term_mode == "full"
    assert decision.long_term_k > 3


def test_execution_repetition_guard_stops_non_progressing_loop():
    class StuckExecutor(ExecutorAgent):
        def __init__(self):
            super().__init__(tool_registry={})

        def is_execution_complete(self, execution_state):
            return False

        def get_runnable_steps(self, execution_state):
            return [{"id": 1, "tool": "noop", "args": {}, "depends_on": []}]

        def run_steps(self, conversation_id, step_index, execution_state, runnable_steps):
            return Message(conversation_id, step_index, "executor", "coordinator", None,
                           "execution_wave_result", "completed", execution_state, "internal")

    coordinator = make_minimal_coordinator(StuckExecutor())
    execution_state = {
        "goal": "stuck",
        "remaining_steps": [{"id": 1, "tool": "noop", "args": {}, "depends_on": []}],
        "step_results": {},
        "step_status": {},
        "execution_trace": [],
        "approved_step_ids": set(),
        "approved_actions": set(),
        "rollback_log": [],
    }

    message = coordinator.run_execution_loop(
        conversation_id=1,
        execution_state=execution_state,
        start_step_index=3,
        user_task="test",
        plan_response={"goal": "stuck", "steps": []},
    )

    assert message.status == "failed"
    assert "repetition guard" in message.response["error"].lower()
