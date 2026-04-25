from typing import Any, Callable
from src.OllamaClient import OllamaClient
from src.tools.file_tools import read_file, list_dir, find_file, create_file, write_file, append_file
from src.tools.llm_tools import create_summarise_txt_func
from src.tools.utils import WorkspaceConfig

def make_tool_registry_entry(func: Callable[..., Any], description: str, input_map: dict[str, str], requires_permission: bool) -> dict[str, Any]:
    return {"func": func, "description": description, "input_map": input_map, "requires_permission": requires_permission}


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
            requires_permission=False
        ),

        "list_dir": make_tool_registry_entry(
            func=lambda path=".": list_dir(workspace, path),
            description="Lists the contents of a directory inside the workspace sandbox.",
            input_map={
                "path": "path",
            },
            requires_permission=False
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
            requires_permission=False
        ),

        "create_file": make_tool_registry_entry(
        func=lambda path, content="": create_file(workspace, path, content),
        description="Creates a new file inside the active workspace sandbox.",
        input_map={
            "path": "path",
            "content": "content"
        },
        requires_permission=True
        ),

        "write_file": make_tool_registry_entry(
            func=lambda path, content: write_file(workspace, path, content),
            description="Writes content to a file inside the active workspace sandbox. This may overwrite existing content.",
            input_map={
                "path": "path",
                "content": "content"
            },
            requires_permission=True
        ),

        "append_file": make_tool_registry_entry(
            func=lambda path, content: append_file(workspace, path, content),
            description="Appends content to an existing or new file inside the active workspace sandbox.",
            input_map={
                "path": "path",
                "content": "content"
            },
            requires_permission=True
        ),
    }