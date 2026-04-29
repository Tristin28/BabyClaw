from typing import Any, Callable
from src.llm.OllamaClient import OllamaClient
from src.tools.file_tools import read_file, list_dir, find_file, create_file, write_file, append_file, rollback_file_snapshot, snapshot_file, delete_file, search_text, replace_text
from src.tools.file_tools import list_tree, find_file_recursive, create_dir, delete_dir, move_path, copy_path, snapshot_path, rollback_path_snapshot, snapshot_many_paths, rollback_many_path_snapshots
from src.tools.llm_tools import create_summarise_txt_func, direct_response, generate_content
from src.tools.utils import WorkspaceConfig

def make_tool_registry_entry(func: Callable[..., Any], description: str, input_map: dict[str, str], requires_permission: bool, 
                             permission_identity_args: list[str] = None,rollback_snapshot: Callable[..., Any] = None, 
                             rollback_apply: Callable[..., Any] = None
                             ) -> dict[str, Any]:
    
    return {"func": func, "description": description, "input_map": input_map, "requires_permission": requires_permission, 
            "permission_identity_args": permission_identity_args or [], "rollback_snapshot": rollback_snapshot, "rollback_apply": rollback_apply}


def build_tool_registry(llm_client: OllamaClient, workspace: WorkspaceConfig) -> dict[str, dict[str, Any]]:
    """
        This method creates the tool registry which is how the executor will interpret the available tools that the agent can apply
        Note desc.. are included only for the system to re-explain what the tools are doing for when someone is reading the code. (i.e. added as like comments)
    """
    summarise_txt = create_summarise_txt_func(llm_client)

    return {
        "read_file": make_tool_registry_entry(
            func=lambda path: read_file(workspace, path), #lambda wraps the tools so executor does not need to know about workspace.
            description="Reads the contents of a text file from inside the workspace sandbox.",
            input_map={
                "path": "path",
            },
            requires_permission=False,
        ),

        "list_dir": make_tool_registry_entry(
            func=lambda path=".": list_dir(workspace, path),
            description="Lists the contents of a directory inside the workspace sandbox.",
            input_map={
                "path": "path",
            },
            requires_permission=False,
        ),

        "find_file": make_tool_registry_entry(
            func=lambda query, directory=".": find_file(workspace, query, directory),
            description="Finds exactly one file in the workspace matching a partial filename query.",
            input_map={
                "query": "query",
                "directory": "directory"
            },
            requires_permission=False
        ),

        "summarise_txt": make_tool_registry_entry(
            func=summarise_txt,
            description="Summarises text produced by a previous step.",
            input_map={
                "text": "text",
            },
            requires_permission=False,
        ),

        "create_file": make_tool_registry_entry(
        func=lambda path, content="": create_file(workspace, path, content),
        description="Creates a new file inside the active workspace sandbox.",
        input_map={
            "path": "path",
            "content": "content"
        },
        requires_permission=True,
        permission_identity_args=["path", "content"],
        rollback_snapshot=lambda path, content="": snapshot_file(workspace, path),
        rollback_apply=lambda snapshot: rollback_file_snapshot(workspace, snapshot)
        ),

        "write_file": make_tool_registry_entry(
            func=lambda path, content: write_file(workspace, path, content),
            description="Writes content to a file inside the active workspace sandbox. This may overwrite existing content.",
            input_map={
                "path": "path",
                "content": "content"
            },
            requires_permission=True,
            permission_identity_args=["path", "content"],
            rollback_snapshot=lambda path, content="": snapshot_file(workspace, path),
            rollback_apply=lambda snapshot: rollback_file_snapshot(workspace, snapshot)
        ),

        "append_file": make_tool_registry_entry(
            func=lambda path, content: append_file(workspace, path, content),
            description="Appends content to an existing or new file inside the active workspace sandbox.",
            input_map={
                "path": "path",
                "content": "content"
            },
            requires_permission=True,
            permission_identity_args=["path", "content"],
            rollback_snapshot=lambda path, content="": snapshot_file(workspace, path),
            rollback_apply=lambda snapshot: rollback_file_snapshot(workspace, snapshot)
        ),

        "direct_response": make_tool_registry_entry(
            func=lambda prompt, context="", recent_messages=None: direct_response(llm_client=llm_client, prompt=prompt, context=context,recent_messages=recent_messages),
            description="Uses the LLM to generate a direct response to the user without modifying files.",
            input_map={
                "prompt": "prompt"
            },
            requires_permission=False
        ),

        "delete_file": make_tool_registry_entry(
            func=lambda path: delete_file(workspace, path),
            description="Deletes a file inside the active workspace sandbox.",
            input_map={
                "path": "path"
            },
            requires_permission=True,
            permission_identity_args=["path"],
            rollback_snapshot=lambda path: snapshot_file(workspace, path),
            rollback_apply=lambda snapshot: rollback_file_snapshot(workspace, snapshot)
        ),

        "search_text": make_tool_registry_entry(
            func=lambda query: search_text(workspace, query),
            description="Searches for text inside files in the workspace sandbox.",
            input_map={
                "query": "query"
            },
            requires_permission=False
        ),

        "replace_text": make_tool_registry_entry(
            func=lambda path, old_text, new_text: replace_text(workspace, path, old_text, new_text),
            description="Replaces text inside a file in the active workspace sandbox.",
            input_map={
                "path": "path",
                "old_text": "old_text",
                "new_text": "new_text"
            },
            requires_permission=True,
            permission_identity_args=["path", "old_text", "new_text"],
            rollback_snapshot=lambda path, old_text, new_text: snapshot_file(workspace, path),
            rollback_apply=lambda snapshot: rollback_file_snapshot(workspace, snapshot)
        ),

        "list_tree": make_tool_registry_entry(
            func=lambda path=".": list_tree(workspace, path, max_depth=5),
            description="Recursively lists files and folders inside a workspace directory.",
            input_map={
                "path": "path"
            },
            requires_permission=False
        ),

        "find_file_recursive": make_tool_registry_entry(
            func=lambda query, directory=".": find_file_recursive(workspace, query, directory),
            description="Recursively finds exactly one file inside a directory and its subdirectories.",
            input_map={
                "query": "query",
                "directory": "directory"
            },
            requires_permission=False
        ),

        "create_dir": make_tool_registry_entry(
            func=lambda path: create_dir(workspace, path),
            description="Creates a directory inside the active workspace sandbox.",
            input_map={
                "path": "path"
            },
            requires_permission=True,
            permission_identity_args=["path"],
            rollback_snapshot=lambda path: snapshot_path(workspace, path),
            rollback_apply=lambda snapshot: rollback_path_snapshot(workspace, snapshot)
        ),

        "delete_dir": make_tool_registry_entry(
            func=lambda path: delete_dir(workspace, path),
            description="Deletes a directory and everything inside it from the active workspace sandbox.",
            input_map={
                "path": "path"
            },
            requires_permission=True,
            permission_identity_args=["path"],
            rollback_snapshot=lambda path: snapshot_path(workspace, path),
            rollback_apply=lambda snapshot: rollback_path_snapshot(workspace, snapshot)
        ),

        "move_path": make_tool_registry_entry(
            func=lambda source_path, destination_path: move_path(workspace, source_path, destination_path),
            description="Moves a file or directory inside the active workspace sandbox.",
            input_map={
                "source_path": "source_path",
                "destination_path": "destination_path"
            },
            requires_permission=True,
            permission_identity_args=["source_path", "destination_path"],
            rollback_snapshot=lambda source_path, destination_path: snapshot_many_paths(
                workspace,
                [source_path, destination_path]
            ),
            rollback_apply=lambda snapshot: rollback_many_path_snapshots(workspace, snapshot)
        ),

        "copy_path": make_tool_registry_entry(
            func=lambda source_path, destination_path: copy_path(workspace, source_path, destination_path),
            description="Copies a file or directory inside the active workspace sandbox.",
            input_map={
                "source_path": "source_path",
                "destination_path": "destination_path"
            },
            requires_permission=True,
            permission_identity_args=["source_path", "destination_path"],
            rollback_snapshot=lambda _, destination_path: snapshot_path(workspace, destination_path),
            rollback_apply=lambda snapshot: rollback_path_snapshot(workspace, snapshot)
        ),

        "generate_content": make_tool_registry_entry(
            func=lambda prompt: generate_content(llm_client=llm_client, prompt=prompt),
            description="Generates raw text/code content using the LLM, intended to be chained into create_file/write_file via content_step.",
            input_map={
                "prompt": "prompt"
            },
            requires_permission=False
        ),
    }