import json

from src.Agents.Reviewing.ReviewerAgent import ReviewerAgent


class FakeReviewerLLM:
    """
    Fake deterministic reviewer LLM.

    This tests the ReviewerAgent wrapper and review evidence flow without
    depending on Ollama/model randomness.
    """

    def invoke_json(self, messages, stream=False, schema=None):
        user_message = messages[-1]["content"]

        task_marker = "CURRENT USER TASK TO REVIEW:"
        evidence_marker = "CLEAN EXECUTION EVIDENCE:"

        task_text = user_message.split(task_marker, 1)[1].split("Important:", 1)[0].strip()
        evidence_text = user_message.split(evidence_marker, 1)[1].strip()
        evidence = json.loads(evidence_text)

        steps = evidence.get("steps", [])

        for step in steps:
            if step.get("status") != "completed":
                return {
                    "accepted": False,
                    "review_summary": "A required tool step failed.",
                    "issues": [f"Step {step.get('id')} failed."]
                }

        mutating_tools = {
            "create_file",
            "write_file",
            "append_file",
            "delete_file",
            "create_dir",
            "delete_dir",
            "move_path",
            "copy_path",
            "replace_text",
        }

        for step in steps:
            tool = step.get("tool")
            args = step.get("resolved_args") or step.get("args", {})

            if tool not in mutating_tools:
                continue

            paths = []

            for key in ("path", "source_path", "destination_path"):
                if key in args:
                    paths.append(args[key])

            for path in paths:
                if path and path not in task_text:
                    return {
                        "accepted": False,
                        "review_summary": "The execution performed an unrequested workspace mutation.",
                        "issues": [f"Unrequested mutation by {tool} on path '{path}'."]
                    }

        return {
            "accepted": True,
            "review_summary": "The execution satisfies the current task.",
            "issues": []
        }


def make_reviewer():
    return ReviewerAgent(llm_client=FakeReviewerLLM())


def test_reviewer_accepts_requested_create_file():
    reviewer = make_reviewer()

    execution_response = {
        "goal": "Create work.txt with hello",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "work.txt", "content": "hello"},
                "resolved_args": {"path": "work.txt", "content": "hello"},
                "result": "File 'work.txt' created successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="create work.txt with hello",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["work.txt"]
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is True
    assert msg.response["issues"] == []


def test_reviewer_rejects_unrequested_mutation():
    reviewer = make_reviewer()

    execution_response = {
        "goal": "Create work.txt with hello",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "work.txt", "content": "hello"},
                "resolved_args": {"path": "work.txt", "content": "hello"},
                "result": "File 'work.txt' created successfully."
            },
            {
                "id": 2,
                "tool": "delete_file",
                "status": "completed",
                "args": {"path": "unrelated.txt"},
                "resolved_args": {"path": "unrelated.txt"},
                "result": "File 'unrelated.txt' deleted successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="create work.txt with hello",
        execution_response=execution_response,
        workspace_before=["unrelated.txt"],
        workspace_after=["work.txt"]
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert any("unrelated.txt" in issue for issue in msg.response["issues"])


def test_reviewer_rejects_wrong_requested_path():
    reviewer = make_reviewer()

    execution_response = {
        "goal": "Create wrong file",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "wrong.txt", "content": "hello"},
                "resolved_args": {"path": "wrong.txt", "content": "hello"},
                "result": "File 'wrong.txt' created successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="create work.txt with hello",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["wrong.txt"]
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert any("wrong.txt" in issue for issue in msg.response["issues"])


def test_reviewer_rejects_failed_tool_step():
    reviewer = make_reviewer()

    execution_response = {
        "goal": "Create work.txt with hello",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "failed",
                "args": {"path": "work.txt", "content": "hello"},
                "resolved_args": {"path": "work.txt", "content": "hello"},
                "error": "File already exists."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="create work.txt with hello",
        execution_response=execution_response,
        workspace_before=["work.txt"],
        workspace_after=["work.txt"]
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert any("failed" in issue.lower() for issue in msg.response["issues"])