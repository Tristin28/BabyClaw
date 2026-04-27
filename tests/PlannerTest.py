import pytest

from src.Agents.Planning.PlanCompiler import PlanCompiler
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.utils import WorkspaceConfig


def make_compiler(tmp_path):
    workspace = WorkspaceConfig(root=str(tmp_path))

    return PlanCompiler(
        available_tools=PLANNER_TOOL_DESCRIPTIONS,
        workspace_config=workspace
    )


def test_valid_content_step_dependency_passes(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Read hello.txt and create BabyClaw with the same content",
        "steps": [
            {
                "id": 1,
                "tool": "read_file",
                "args": {"path": "hello.txt"}
            },
            {
                "id": 2,
                "tool": "create_file",
                "args": {
                    "path": "BabyClaw",
                    "content_step": 1
                }
            }
        ],
        "planning_rationale": "Read the source file, then create the new file using that content."
    }

    compiled = compiler.compile(plan)

    assert compiled["goal"] == plan["goal"]
    assert compiled["steps"][0]["tool"] == "read_file"
    assert compiled["steps"][1]["tool"] == "create_file"
    assert compiled["steps"][1]["args"]["content_step"] == 1
    assert compiled["steps"][1]["depends_on"] == [1]


def test_planner_unknown_tool_rejected(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Use a hallucinated tool",
        "steps": [
            {
                "id": 1,
                "tool": "hack_shell",
                "args": {"command": "rm -rf /"}
            }
        ],
        "planning_rationale": "Bad hallucinated plan."
    }

    with pytest.raises(ValueError, match="unknown tool"):
        compiler.compile(plan)


def test_planner_unknown_argument_rejected(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Read a file with a hallucinated argument name",
        "steps": [
            {
                "id": 1,
                "tool": "read_file",
                "args": {"filename": "hello.txt"}
            }
        ],
        "planning_rationale": "Bad argument name."
    }

    with pytest.raises(ValueError, match="unknown arg"):
        compiler.compile(plan)


def test_fake_step_string_rejected(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Create a file using fake step syntax",
        "steps": [
            {
                "id": 1,
                "tool": "read_file",
                "args": {"path": "hello.txt"}
            },
            {
                "id": 2,
                "tool": "create_file",
                "args": {
                    "path": "BabyClaw",
                    "content": "text_step:1"
                }
            }
        ],
        "planning_rationale": "Bad fake step string."
    }

    with pytest.raises(ValueError):
        compiler.compile(plan)


def test_future_step_reference_rejected(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Use a future step result",
        "steps": [
            {
                "id": 1,
                "tool": "summarise_txt",
                "args": {"text_step": 2}
            },
            {
                "id": 2,
                "tool": "read_file",
                "args": {"path": "hello.txt"}
            }
        ],
        "planning_rationale": "Bad future dependency."
    }

    with pytest.raises(ValueError):
        compiler.compile(plan)


def test_path_traversal_rejected_before_execution(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Try to read outside workspace",
        "steps": [
            {
                "id": 1,
                "tool": "read_file",
                "args": {"path": "../../etc/passwd"}
            }
        ],
        "planning_rationale": "Unsafe path."
    }

    with pytest.raises(ValueError, match="outside the workspace|unsafe"):
        compiler.compile(plan)


def test_absolute_path_rejected_before_execution(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Try absolute path",
        "steps": [
            {
                "id": 1,
                "tool": "read_file",
                "args": {"path": "/etc/passwd"}
            }
        ],
        "planning_rationale": "Unsafe absolute path."
    }

    with pytest.raises(ValueError, match="relative workspace path"):
        compiler.compile(plan)


def test_mutating_tool_cannot_target_workspace_root(tmp_path):
    compiler = make_compiler(tmp_path)

    plan = {
        "goal": "Try to delete the workspace root",
        "steps": [
            {
                "id": 1,
                "tool": "delete_dir",
                "args": {"path": "."}
            }
        ],
        "planning_rationale": "Unsafe mutation."
    }

    with pytest.raises(ValueError, match="workspace root"):
        compiler.compile(plan)