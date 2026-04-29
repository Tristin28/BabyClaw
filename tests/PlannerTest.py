import pytest

from src.agents.planning.PlanCompiler import PlanCompiler
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS
from src.tools.utils import WorkspaceConfig


def make_compiler(tmp_path):
    workspace = WorkspaceConfig(root=str(tmp_path))

    return PlanCompiler(
        available_tools=PLANNER_TOOL_DESCRIPTIONS,
        workspace_config=workspace
    )


def make_compiler_for_task(tmp_path, user_task):
    workspace = WorkspaceConfig(root=str(tmp_path))

    return PlanCompiler(
        available_tools=PLANNER_TOOL_DESCRIPTIONS,
        workspace_config=workspace,
        route={"task_type": "workspace_mutation"},
        user_task=user_task
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


def test_bare_content_step_string_repaired_when_single_generate_content_source(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create src/main.py with a simple Python hello world program"
    )

    plan = {
        "goal": "Create hello world program",
        "steps": [
            {
                "id": 1,
                "tool": "generate_content",
                "args": {
                    "prompt": "Write a simple Python hello world program. Output only Python code."
                }
            },
            {
                "id": 2,
                "tool": "create_file",
                "args": {
                    "path": "src/main.py",
                    "content": "content_step"
                }
            }
        ],
        "planning_rationale": "Bad bare step marker."
    }

    compiled = compiler.compile(plan)

    assert "content" not in compiled["steps"][1]["args"]
    assert compiled["steps"][1]["args"]["content_step"] == 1
    assert compiled["steps"][1]["depends_on"] == [1]


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


def test_explicit_requested_file_path_rejects_unrelated_mutation_path(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Write a poem about Malta into poem.txt"
    )

    plan = {
        "goal": "Write poem into a file",
        "steps": [
            {
                "id": 1,
                "tool": "create_dir",
                "args": {"path": "documents"}
            },
            {
                "id": 2,
                "tool": "create_file",
                "args": {
                    "path": "documents/inspirational_quotes.txt",
                    "content": "Malta poem"
                }
            }
        ],
        "planning_rationale": "Bad path drift."
    }

    with pytest.raises(ValueError, match="planned mutation path 'documents' does not match requested path 'poem.txt'"):
        compiler.compile(plan)


def test_explicit_requested_file_path_allows_exact_mutation_path(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Write a poem about Malta into poem.txt"
    )

    plan = {
        "goal": "Write poem into requested file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "poem.txt",
                    "content": "Malta poem"
                }
            }
        ],
        "planning_rationale": "Use the requested path."
    }

    compiled = compiler.compile(plan)

    assert compiled["steps"][0]["args"]["path"] == "poem.txt"


def test_explicit_nested_file_path_allows_parent_directory_and_exact_file(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create src/main.py with hello world"
    )

    plan = {
        "goal": "Create nested requested file",
        "steps": [
            {
                "id": 1,
                "tool": "create_dir",
                "args": {"path": "src"}
            },
            {
                "id": 2,
                "tool": "create_file",
                "args": {
                    "path": "src/main.py",
                    "content": "print('hello world')"
                }
            }
        ],
        "planning_rationale": "Create parent and file."
    }

    compiled = compiler.compile(plan)

    assert compiled["steps"][0]["args"]["path"] == "src"
    assert compiled["steps"][1]["args"]["path"] == "src/main.py"


def test_explicit_nested_file_path_rejects_wrong_directory(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create src/main.py with hello world"
    )

    plan = {
        "goal": "Create wrong nested file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "app/main.py",
                    "content": "print('hello world')"
                }
            }
        ],
        "planning_rationale": "Bad path drift."
    }

    with pytest.raises(ValueError, match="planned mutation path 'app/main.py' does not match requested path 'src/main.py'"):
        compiler.compile(plan)


def test_explicit_nested_file_path_rejects_code_py(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create src/main.py with a simple Python hello world program"
    )

    plan = {
        "goal": "Create wrong file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "code.py",
                    "content": "print('hello world')"
                }
            }
        ],
        "planning_rationale": "Bad inferred filename."
    }

    with pytest.raises(ValueError, match="planned mutation path 'code.py' does not match requested path 'src/main.py'"):
        compiler.compile(plan)


def test_explicit_nested_file_path_rejects_hello_world_py(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create src/main.py with a simple Python hello world program"
    )

    plan = {
        "goal": "Create wrong file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "hello_world.py",
                    "content": "print('hello world')"
                }
            }
        ],
        "planning_rationale": "Bad inferred filename."
    }

    with pytest.raises(ValueError, match="planned mutation path 'hello_world.py' does not match requested path 'src/main.py'"):
        compiler.compile(plan)


def test_quoted_explicit_file_path_with_spaces_is_preserved(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        'Create "my notes.txt" with hello'
    )

    plan = {
        "goal": "Create quoted file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "my notes.txt",
                    "content": "hello"
                }
            }
        ],
        "planning_rationale": "Use quoted filename."
    }

    compiled = compiler.compile(plan)

    assert compiled["steps"][0]["args"]["path"] == "my notes.txt"


def test_multiple_explicit_paths_allow_move_source_and_destination(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Rename draft.txt to final.txt"
    )

    plan = {
        "goal": "Rename requested file",
        "steps": [
            {
                "id": 1,
                "tool": "move_path",
                "args": {
                    "source_path": "draft.txt",
                    "destination_path": "final.txt"
                }
            }
        ],
        "planning_rationale": "Move source to destination."
    }

    compiled = compiler.compile(plan)

    assert compiled["steps"][0]["args"]["source_path"] == "draft.txt"
    assert compiled["steps"][0]["args"]["destination_path"] == "final.txt"


def test_exact_write_text_is_enforced_when_planner_substitutes_content(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create a text file and write Tristin inside it"
    )

    plan = {
        "goal": "Create generic file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "test.txt",
                    "content": "Hello, world!"
                }
            }
        ],
        "planning_rationale": "Bad content drift."
    }

    compiled = compiler.compile(plan)

    assert compiled["steps"][0]["args"]["path"] == "test.txt"
    assert compiled["steps"][0]["args"]["content"] == "Tristin"


def test_exact_write_text_is_enforced_for_inside_it_write_order(tmp_path):
    compiler = make_compiler_for_task(
        tmp_path,
        "Create a text file and inside it write Tristin"
    )

    plan = {
        "goal": "Create hello world file",
        "steps": [
            {
                "id": 1,
                "tool": "create_file",
                "args": {
                    "path": "hello_world.txt",
                    "content": "Hello, World!"
                }
            }
        ],
        "planning_rationale": "Bad content drift."
    }

    compiled = compiler.compile(plan)

    assert compiled["steps"][0]["args"]["path"] == "hello_world.txt"
    assert compiled["steps"][0]["args"]["content"] == "Tristin"
