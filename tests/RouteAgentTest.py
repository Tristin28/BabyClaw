from src.agents.routing.RouteAgent import RouteAgent
from src.agents.routing.WorkflowPolicy import WorkflowPolicyRegistry
import pytest


class FakeRouterLLM:
    def __init__(self, task_type: str):
        self.task_type = task_type

    def invoke_json(self, messages, stream=False, schema=None):
        return {
            "task_type": self.task_type,
            "confidence": 1.0,
            "routing_reason": "fake route"
        }


def test_router_trusts_llm_workspace_mutation_classification():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="workspace_mutation"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write me a short story about a robot in a text file"
    )

    assert msg.response["task_type"] == "workspace_mutation"


@pytest.mark.parametrize("user_task", [
    "write me an email to my friend",
    "write me a description I can add in my bio",
    "write a Python function",
    "create a short paragraph about AI",
    "summarise this text",
])
def test_router_downgrades_workspace_mutation_without_explicit_workspace_side_effect(user_task):
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="workspace_mutation"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task=user_task
    )

    assert msg.response["task_type"] == "direct_response"
    assert "did not explicitly request" in msg.response["routing_reason"].lower()


@pytest.mark.parametrize("user_task", [
    "write this to bio.txt",
    "create a file called bio.txt",
    "append this to notes.md",
    "delete old_report.txt",
    "rename draft.md to final_draft.md",
    "save this inside the workspace",
])
def test_router_allows_workspace_mutation_with_explicit_workspace_side_effect(user_task):
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="workspace_mutation"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task=user_task
    )

    assert msg.response["task_type"] == "workspace_mutation"


def test_router_trusts_llm_contextual_followup_classification():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="contextual_followup"))

    examples = [
        "Write another version",
        "Continue it",
        "Make it shorter",
        "Do the same for Jake",
        "Use the previous result",
    ]

    for user_task in examples:
        msg = router.run(
            conversation_id=1,
            step_index=1,
            user_task=user_task
        )

        assert msg.response["task_type"] == "contextual_followup"


def test_router_trusts_llm_direct_response_classification():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="direct_response"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a short story about a robot learning friendship"
    )

    assert msg.response["task_type"] == "direct_response"


def test_router_forces_direct_response_for_in_chat_requests():
    """
    The only deterministic override left: explicit chat requests must never
    be turned into a file mutation, even if the LLM picked workspace_mutation.
    """
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="workspace_mutation"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a poem in chat"
    )

    assert msg.response["task_type"] == "direct_response"
    assert "in chat" in msg.response["routing_reason"].lower()


def test_router_does_not_upgrade_to_mutation_when_llm_says_direct_response():
    """
    Heavy keyword override that upgraded "in a file" phrases to
    workspace_mutation has been removed. The LLM is now responsible for
    that classification.
    """
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="direct_response"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a poem in a text file"
    )

    assert msg.response["task_type"] == "direct_response"


def test_direct_response_policy_does_not_use_memory_or_recent_messages_by_default():
    route = WorkflowPolicyRegistry.build_route({
        "task_type": "direct_response",
        "confidence": 1.0,
        "routing_reason": "standalone chat"
    })

    assert route["memory_mode"] == "none"
    assert route["use_recent_messages"] is False
