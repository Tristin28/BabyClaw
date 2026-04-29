import json

from src.agents.reviewing.ReviewerAgent import ReviewerAgent
from src.tools.utils import WorkspaceConfig


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


def test_reviewer_accepts_valid_direct_response_without_llm_overreview():
    class RejectingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            raise AssertionError("Direct response review should not call the LLM reviewer")

    reviewer = ReviewerAgent(llm_client=RejectingReviewerLLM())

    execution_response = {
        "goal": "direct_response",
        "execution_trace": [
            {
                "id": 1,
                "tool": "direct_response",
                "status": "completed",
                "args": {"prompt": "Write a message to my friend Jake and ask how he is"},
                "resolved_args": {"prompt": "Write a message to my friend Jake and ask how he is"},
                "result": "Hi Jake,\n\nHow are you doing? Hope you're well.\n\nTristin"
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Write a message to my friend Jake and ask how he is",
        execution_response=execution_response,
        workspace_before=["poem.txt"],
        workspace_after=["poem.txt"],
        route={
            "task_type": "contextual_followup",
            "tool_group": "direct_response_tools",
            "use_workspace": False,
            "allow_mutations": False,
            "allowed_tools": ["direct_response"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is True
    assert msg.response["issues"] == []


def test_reviewer_rejects_failed_direct_response_without_llm_overreview():
    class RejectingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            raise AssertionError("Direct response review should not call the LLM reviewer")

    reviewer = ReviewerAgent(llm_client=RejectingReviewerLLM())

    execution_response = {
        "goal": "direct_response",
        "execution_trace": [
            {
                "id": 1,
                "tool": "direct_response",
                "status": "failed",
                "args": {"prompt": "Write a message to my friend Jake and ask how he is"},
                "resolved_args": {"prompt": "Write a message to my friend Jake and ask how he is"},
                "error": "LLM failed"
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Write a message to my friend Jake and ask how he is",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=[],
        route={
            "task_type": "contextual_followup",
            "tool_group": "direct_response_tools",
            "use_workspace": False,
            "allow_mutations": False,
            "allowed_tools": ["direct_response"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert any("complete successfully" in issue for issue in msg.response["issues"])


class SemanticReviewerLLM:
    def invoke_json(self, messages, stream=False, schema=None):
        user_message = messages[-1]["content"]

        task_marker = "CURRENT USER TASK TO REVIEW:"
        evidence_marker = "CLEAN EXECUTION EVIDENCE:"

        task_text = user_message.split(task_marker, 1)[1].split("Important:", 1)[0].strip().lower()
        evidence_text = user_message.split(evidence_marker, 1)[1].strip()
        evidence = json.loads(evidence_text)

        final_contents = []

        for step in evidence.get("steps", []):
            final_content = step.get("final_content", {})

            if final_content.get("available"):
                final_contents.append(final_content.get("content", ""))

        joined_content = "\n".join(final_contents).lower()

        if "snake game" in task_text and "hello" in joined_content and "game" not in joined_content:
            return {
                "accepted": False,
                "review_summary": "The file was created, but its content does not meaningfully match the requested artifact.",
                "issues": ["The saved Python file contains hello-world style content instead of a meaningful snake game implementation."]
            }

        return {
            "accepted": True,
            "review_summary": "The created artifact meaningfully satisfies the current task.",
            "issues": []
        }


def test_reviewer_includes_final_file_content_for_semantic_review(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "game.py"
    target_file.write_text("def hello_world():\n    print('Hello world')\n", encoding="utf-8")

    reviewer = ReviewerAgent(
        llm_client=SemanticReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Create a snake game inside a Python file",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "game.py", "content": "def hello_world():\n    print('Hello world')\n"},
                "resolved_args": {"path": "game.py", "content": "def hello_world():\n    print('Hello world')\n"},
                "result": "File 'game.py' created successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Create me a snake game inside a Python file",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["game.py"]
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert any("snake game" in issue.lower() for issue in msg.response["issues"])


def test_reviewer_accepts_hello_world_when_requested(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "hello.py"
    target_file.write_text("def hello_world():\n    print('Hello world')\n", encoding="utf-8")

    reviewer = ReviewerAgent(
        llm_client=SemanticReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Create a simple hello world Python file",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "hello.py", "content": "def hello_world():\n    print('Hello world')\n"},
                "resolved_args": {"path": "hello.py", "content": "def hello_world():\n    print('Hello world')\n"},
                "result": "File 'hello.py' created successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Create a simple hello world Python file",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["hello.py"]
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is True
    assert msg.response["issues"] == []


def test_reviewer_rejects_written_file_when_proper_noun_topic_is_missing(tmp_path):
    class RejectingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            raise AssertionError("Deterministic topic check should reject before the LLM reviewer")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "poem.txt"
    target_file.write_text(
        "# A Simple Poem\n\n"
        "In a meadow where flowers bloom,\n"
        "A butterfly dances, free and bold...\n",
        encoding="utf-8"
    )

    reviewer = ReviewerAgent(
        llm_client=RejectingReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Write a poem about Malta into poem.txt",
        "execution_trace": [
            {
                "id": 1,
                "tool": "write_file",
                "status": "completed",
                "args": {"path": "poem.txt", "content": target_file.read_text(encoding="utf-8")},
                "resolved_args": {"path": "poem.txt", "content": target_file.read_text(encoding="utf-8")},
                "result": "File 'poem.txt' written successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Write a poem about Malta into poem.txt",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["poem.txt"],
        route={
            "task_type": "workspace_mutation",
            "tool_group": "mutation_file_tools",
            "use_workspace": True,
            "allow_mutations": True,
            "allowed_tools": ["write_file"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert msg.response["issues"] == [
        "The file was written, but the content does not match the requested topic 'Malta'."
    ]


def test_reviewer_allows_written_file_when_proper_noun_topic_is_present(tmp_path):
    class AcceptingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            return {
                "accepted": True,
                "review_summary": "The file content matches the requested topic.",
                "issues": []
            }

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "poem.txt"
    target_file.write_text(
        "Malta wakes where honeyed limestone glows,\n"
        "And waves remember every harbour light.\n",
        encoding="utf-8"
    )

    reviewer = ReviewerAgent(
        llm_client=AcceptingReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Write a poem about Malta into poem.txt",
        "execution_trace": [
            {
                "id": 1,
                "tool": "write_file",
                "status": "completed",
                "args": {"path": "poem.txt", "content": target_file.read_text(encoding="utf-8")},
                "resolved_args": {"path": "poem.txt", "content": target_file.read_text(encoding="utf-8")},
                "result": "File 'poem.txt' written successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Write a poem about Malta into poem.txt",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["poem.txt"],
        route={
            "task_type": "workspace_mutation",
            "tool_group": "mutation_file_tools",
            "use_workspace": True,
            "allow_mutations": True,
            "allowed_tools": ["write_file"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is True
    assert msg.response["issues"] == []


def test_reviewer_accepts_exact_hello_world_written_to_file(tmp_path):
    class AcceptingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            return {
                "accepted": True,
                "review_summary": "The exact requested content was written.",
                "issues": []
            }

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "hello.txt"
    target_file.write_text("hello world", encoding="utf-8")

    reviewer = ReviewerAgent(
        llm_client=AcceptingReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Write hello world into hello.txt",
        "execution_trace": [
            {
                "id": 1,
                "tool": "write_file",
                "status": "completed",
                "args": {"path": "hello.txt", "content": "hello world"},
                "resolved_args": {"path": "hello.txt", "content": "hello world"},
                "result": "File 'hello.txt' written successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Write hello world into hello.txt",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["hello.txt"],
        route={
            "task_type": "workspace_mutation",
            "tool_group": "mutation_file_tools",
            "use_workspace": True,
            "allow_mutations": True,
            "allowed_tools": ["write_file"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is True
    assert msg.response["issues"] == []


def test_reviewer_rejects_wrong_exact_text_written_inside_unspecified_file(tmp_path):
    class RejectingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            raise AssertionError("Deterministic exact text check should reject before the LLM reviewer")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "test.txt"
    target_file.write_text("Hello, world!", encoding="utf-8")

    reviewer = ReviewerAgent(
        llm_client=RejectingReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Create a text file and write Tristin inside it",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "test.txt", "content": "Hello, world!"},
                "resolved_args": {"path": "test.txt", "content": "Hello, world!"},
                "result": "File 'test.txt' created successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Create a text file and write Tristin inside it",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["test.txt"],
        route={
            "task_type": "workspace_mutation",
            "tool_group": "mutation_file_tools",
            "use_workspace": True,
            "allow_mutations": True,
            "allowed_tools": ["create_file"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert msg.response["issues"] == [
        "The file was written, but the content does not contain the requested text 'Tristin'."
    ]


def test_reviewer_rejects_wrong_exact_text_for_inside_it_write_order(tmp_path):
    class RejectingReviewerLLM:
        def invoke_json(self, messages, stream=False, schema=None):
            raise AssertionError("Deterministic exact text check should reject before the LLM reviewer")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target_file = workspace_root / "hello_world.txt"
    target_file.write_text("Hello, World!", encoding="utf-8")

    reviewer = ReviewerAgent(
        llm_client=RejectingReviewerLLM(),
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    )

    execution_response = {
        "goal": "Create a text file and inside it write Tristin",
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "args": {"path": "hello_world.txt", "content": "Hello, World!"},
                "resolved_args": {"path": "hello_world.txt", "content": "Hello, World!"},
                "result": "File 'hello_world.txt' created successfully."
            }
        ]
    }

    msg = reviewer.run(
        conversation_id=1,
        step_index=3,
        user_task="Create a text file and inside it write Tristin",
        execution_response=execution_response,
        workspace_before=[],
        workspace_after=["hello_world.txt"],
        route={
            "task_type": "workspace_mutation",
            "tool_group": "mutation_file_tools",
            "use_workspace": True,
            "allow_mutations": True,
            "allowed_tools": ["create_file"],
        }
    )

    assert msg.status == "completed"
    assert msg.response["accepted"] is False
    assert msg.response["issues"] == [
        "The file was written, but the content does not contain the requested text 'Tristin'."
    ]
