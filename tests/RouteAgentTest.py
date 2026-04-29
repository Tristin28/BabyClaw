from src.agents.routing.RouteAgent import RouteAgent
from src.agents.routing.WorkflowPolicy import WorkflowPolicyRegistry


class FakeRouterLLM:
    def __init__(self, task_type: str):
        self.task_type = task_type

    def invoke_json(self, messages, stream=False, schema=None):
        return {
            "task_type": self.task_type,
            "confidence": 1.0,
            "routing_reason": "fake route"
        }


def test_router_normalises_standalone_creative_writing_to_direct_response():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="contextual_followup"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a poem in chat"
    )

    assert msg.response["task_type"] == "direct_response"
    assert "Standalone chat" in msg.response["routing_reason"]


def test_router_keeps_explicit_followup_as_contextual_followup():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="contextual_followup"))

    examples = [
        "Write another poem",
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


def test_direct_response_policy_does_not_use_memory_or_recent_messages_by_default():
    route = WorkflowPolicyRegistry.build_route({
        "task_type": "direct_response",
        "confidence": 1.0,
        "routing_reason": "standalone chat"
    })

    assert route["memory_mode"] == "none"
    assert route["use_recent_messages"] is False


def test_story_in_text_file_routes_to_workspace_mutation():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="direct_response"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write me a short story about a robot learning friendship in a text file"
    )

    assert msg.response["task_type"] == "workspace_mutation"


def test_story_without_file_phrase_routes_to_direct_response():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="direct_response"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a short story about a robot learning friendship"
    )

    assert msg.response["task_type"] == "direct_response"


def test_story_in_chat_routes_to_direct_response():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="direct_response"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a short story about a robot learning friendship in chat"
    )

    assert msg.response["task_type"] == "direct_response"


def test_poem_in_text_file_routes_to_workspace_mutation():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="direct_response"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a poem in a text file"
    )

    assert msg.response["task_type"] == "workspace_mutation"


def test_poem_in_chat_routes_to_direct_response():
    router = RouteAgent(llm_client=FakeRouterLLM(task_type="contextual_followup"))

    msg = router.run(
        conversation_id=1,
        step_index=1,
        user_task="Write a poem in chat"
    )

    assert msg.response["task_type"] == "direct_response"
