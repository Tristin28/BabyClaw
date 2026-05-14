"""
    Safety and context tests for BabyClaw.

    Covers:
        - workspace sandbox boundaries (absolute, traversal, symlink escape)
        - route/tool scope guarantees enforced by the executor
        - memory routing for direct_response and contextual_followup
        - filename hints / workspace tree wiring in the planner input

    These tests deliberately avoid any LLM call. They exercise the
    deterministic safety layer that should hold regardless of model output.
"""
import os
from pathlib import Path

import pytest

from src.action_constants import MUTATION_FILE_TOOLS, READ_FILE_TOOLS, SUMMARISE_FILE_TOOLS
from src.agents.execution.ExecutorAgent import ExecutorAgent
from src.agents.routing.MemoryRoutingPolicy import MemoryRoutingPolicy
from src.agents.routing.WorkflowPolicy import WorkflowPolicyRegistry
from src.core.context.ActiveContext import ActiveContext
from src.core.workflow.Coordinator import Coordinator
from src.core.message import Message
from src.tools.file_tools import (
    create_file,
    delete_file,
    move_path,
    replace_text,
    write_file,
)
from src.tools.utils import WorkspaceConfig


# ---------------------------------------------------------------------------
# Workspace sandbox safety
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    return WorkspaceConfig(root=str(tmp_path))


def test_resolve_rejects_relative_traversal(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve_workspace_path("../outside.txt")


def test_resolve_rejects_nested_traversal(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve_workspace_path("../../outside.txt")


def test_resolve_rejects_absolute_unix_path(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve_workspace_path("/absolute/path/outside.txt")


def test_resolve_rejects_absolute_windows_path(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve_workspace_path("C:/Windows/outside.txt")


def test_resolve_rejects_empty_path(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve_workspace_path("")


def test_resolve_rejects_symlink_escape(tmp_path):
    """
        A symlink inside the workspace that points outside must never be
        readable, writable, deletable, or movable through the file tools.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    # Symlink inside the workspace pointing at a directory outside.
    escape_link = workspace_root / "escape"
    escape_link.symlink_to(outside)

    workspace = WorkspaceConfig(root=str(workspace_root))

    with pytest.raises(PermissionError):
        workspace.resolve_workspace_path("escape/secret.txt")

    # File tools must refuse just as firmly.
    with pytest.raises(PermissionError):
        delete_file(workspace, "escape/secret.txt")

    with pytest.raises(PermissionError):
        write_file(workspace, "escape/secret.txt", "tampered")

    # Outside file still intact.
    assert (outside / "secret.txt").read_text() == "classified"


def test_delete_file_refuses_workspace_root(workspace, tmp_path):
    # delete_file on "." or "./" goes through resolve_workspace_path but the
    # underlying file_tools.delete_file refuses non-files. We assert no harm.
    with pytest.raises(Exception):
        delete_file(workspace, ".")

    with pytest.raises(Exception):
        delete_file(workspace, "./")

    # Workspace folder itself is intact.
    assert tmp_path.exists() and tmp_path.is_dir()


def test_move_path_blocks_destination_outside_workspace(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceConfig(root=str(workspace_root))

    src = workspace_root / "inside.txt"
    src.write_text("hello")

    with pytest.raises(PermissionError):
        move_path(workspace, "inside.txt", "../escaped.txt")


def test_replace_text_blocks_paths_outside_workspace(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = WorkspaceConfig(root=str(workspace_root))

    with pytest.raises(PermissionError):
        replace_text(workspace, "/etc/hosts", "old", "new")


# ---------------------------------------------------------------------------
# Route / tool scope safety
# ---------------------------------------------------------------------------

def test_direct_response_route_only_exposes_direct_response_tool():
    route = WorkflowPolicyRegistry.build_route({"task_type": "direct_response",
                                                "confidence": 1.0,
                                                "routing_reason": "x"})

    assert set(route["allowed_tools"]) == {"direct_response"}
    assert route["allow_mutations"] is False


def test_workspace_read_route_does_not_include_mutation_tools():
    route = WorkflowPolicyRegistry.build_route({"task_type": "workspace_read",
                                                "confidence": 1.0,
                                                "routing_reason": "x"})

    forbidden = MUTATION_FILE_TOOLS - READ_FILE_TOOLS - SUMMARISE_FILE_TOOLS - {"summarise_txt"}
    mutation_only = {"create_file", "write_file", "append_file", "delete_file",
                     "create_dir", "delete_dir", "move_path", "copy_path",
                     "replace_text"}

    for tool in mutation_only:
        assert tool not in route["allowed_tools"], f"{tool} leaked into workspace_read"
    assert route["allow_mutations"] is False


def test_workspace_summarise_route_does_not_include_mutation_tools():
    route = WorkflowPolicyRegistry.build_route({"task_type": "workspace_summarise",
                                                "confidence": 1.0,
                                                "routing_reason": "x"})

    mutation_only = {"create_file", "write_file", "append_file", "delete_file",
                     "create_dir", "delete_dir", "move_path", "copy_path",
                     "replace_text"}

    for tool in mutation_only:
        assert tool not in route["allowed_tools"], f"{tool} leaked into workspace_summarise"
    assert route["allow_mutations"] is False


def test_workspace_mutation_route_includes_mutation_tools_and_requires_permission_at_runtime():
    route = WorkflowPolicyRegistry.build_route({"task_type": "workspace_mutation",
                                                "confidence": 1.0,
                                                "routing_reason": "x"})

    for tool in {"create_file", "write_file", "delete_file", "move_path"}:
        assert tool in route["allowed_tools"]
    assert route["allow_mutations"] is True


def test_executor_rejects_tools_outside_allowed_tools():
    """
    Even if a malformed plan slips past PlanCompiler, the executor must
    refuse tools that are not in route.allowed_tools.
    """
    executor = ExecutorAgent(tool_registry={"create_file": {"func": lambda **kw: "ok",
                                                            "input_map": {"path": "path"},
                                                            "requires_permission": False}})

    execution_state = {
        "allowed_tools": {"direct_response"},
        "allow_mutations": False,
    }

    with pytest.raises(ValueError, match="outside the allowed route scope"):
        executor.validate_step_scope(
            step={"tool": "create_file", "args": {"path": "x"}},
            execution_state=execution_state,
        )


def test_executor_rejects_mutation_when_allow_mutations_is_false():
    executor = ExecutorAgent(tool_registry={})

    execution_state = {
        "allowed_tools": {"create_file"},
        "allow_mutations": False,
    }

    with pytest.raises(ValueError, match="does not allow mutations"):
        executor.validate_step_scope(
            step={"tool": "create_file", "args": {"path": "x"}},
            execution_state=execution_state,
        )


# ---------------------------------------------------------------------------
# Memory routing context behaviour
# ---------------------------------------------------------------------------

def test_what_is_reinforcement_learning_does_not_load_memory():
    decision = MemoryRoutingPolicy.decide(
        user_task="what is reinforcement learning?",
        route={"task_type": "direct_response"},
    )

    assert decision.use_long_term is False
    assert decision.use_short_term is False
    assert decision.long_term_mode == "none"


def test_what_is_it_used_for_does_not_load_memory_under_direct_response():
    """
    Bare pronouns no longer trigger memory loading. Genuine follow-ups go
    through contextual_followup classification instead.
    """
    decision = MemoryRoutingPolicy.decide(
        user_task="what is it used for?",
        route={"task_type": "direct_response"},
    )

    assert decision.use_long_term is False
    assert decision.use_short_term is False


def test_contextual_followup_loads_short_term_context():
    decision = MemoryRoutingPolicy.decide(
        user_task="make it shorter",
        route={"task_type": "contextual_followup"},
    )

    assert decision.use_short_term is True
    assert decision.short_term_k > 0
    assert decision.use_long_term is True


def test_continue_keyword_still_pulls_memory_under_direct_response():
    decision = MemoryRoutingPolicy.decide(
        user_task="continue from earlier",
        route={"task_type": "direct_response"},
    )

    assert decision.use_long_term is True


# ---------------------------------------------------------------------------
# Coordinator filename hints + workspace context wiring
# ---------------------------------------------------------------------------

class _StubCoordinator(Coordinator):
    """
    Build a Coordinator skeleton just for `build_filename_hints` and
    `build_scoped_workspace_contents`. Nothing else is exercised.
    """

    def __init__(self, tool_registry, active_context=None):
        # Skip the parent constructor and only set fields the methods use.
        self.tool_registry = tool_registry
        self.active_context = active_context or ActiveContext()
        self.workspace_config = None
        self.planner_tool_descriptions = []


def test_filename_hints_resolve_bare_name_to_unique_match():
    coordinator = _StubCoordinator(tool_registry={})
    workspace_contents = [
        "Coordinator.py",
        "src/",
        "src/notes.txt",
        "src/utils.py",
    ]

    hints = coordinator.build_filename_hints(
        user_task="open Coordinator",
        workspace_contents=workspace_contents,
        route={"task_type": "workspace_read"},
    )

    assert hints.get("Coordinator") == "Coordinator.py"


def test_filename_hints_surface_ambiguity_for_read_routes():
    coordinator = _StubCoordinator(tool_registry={})
    workspace_contents = [
        "Coordinator.py",
        "src/Coordinator.md",
    ]

    hints = coordinator.build_filename_hints(
        user_task="open Coordinator",
        workspace_contents=workspace_contents,
        route={"task_type": "workspace_read"},
    )

    assert isinstance(hints.get("Coordinator"), list)
    assert set(hints["Coordinator"]) == {"Coordinator.py", "src/Coordinator.md"}


def test_filename_hints_hide_ambiguity_for_mutation_routes():
    """
    Ambiguous filename hints must not be surfaced to the planner on
    mutation routes; otherwise it might pick the wrong file.
    """
    coordinator = _StubCoordinator(tool_registry={})
    workspace_contents = [
        "Coordinator.py",
        "src/Coordinator.md",
    ]

    hints = coordinator.build_filename_hints(
        user_task="delete Coordinator",
        workspace_contents=workspace_contents,
        route={"task_type": "workspace_mutation"},
    )

    assert "Coordinator" not in hints


def test_workspace_view_included_for_contextual_followup_with_active_file():
    active_context = ActiveContext()
    active_context.record_file_modified("notes.txt")

    tool_registry = {
        "list_tree": {"func": lambda path=".": ["notes.txt", "src/"]}
    }

    coordinator = _StubCoordinator(tool_registry=tool_registry, active_context=active_context)

    contents = coordinator.build_scoped_workspace_contents(route={
        "task_type": "contextual_followup",
        "use_workspace": False,
    })

    assert contents == ["notes.txt", "src/"]


def test_workspace_view_skipped_for_direct_response_without_active_file():
    coordinator = _StubCoordinator(
        tool_registry={"list_tree": {"func": lambda path=".": ["should_not_be_returned"]}},
    )

    contents = coordinator.build_scoped_workspace_contents(route={
        "task_type": "direct_response",
        "use_workspace": False,
    })

    assert contents == []
