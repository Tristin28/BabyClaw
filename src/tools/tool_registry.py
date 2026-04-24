from typing import Any, Callable
from src.OllamaClient import OllamaClient
from src.tools.file_tools import read_file, list_dir
from src.tools.llm_tools import create_summarise_txt_func


def make_tool_registry_entry(func: Callable[..., Any], description: str, input_map: dict[str, str], requires_permission: bool) -> dict[str, Any]:
    return {"func": func, "description": description, "input_map": input_map, "requires_permission": requires_permission}


def build_tool_registry(llm_client: OllamaClient) -> dict[str, dict[str, Any]]:
    """
        This method creates the tool registry which is how the executor will interpret the available tools that the agent can apply
        Note desc.. are included only for the system to re-explain what the tools are doing for when someone is reading the code. (i.e. added as like comments)
    """
    summarise_txt = create_summarise_txt_func(llm_client)

    return {
        "read_file": make_tool_registry_entry(
            func=read_file,
            description="Reads the contents of a text file from inside the workspace sandbox.",
            input_map={
                "path": "path",
            },
            requires_permission=False
        ),
        "list_dir": make_tool_registry_entry(
            func=list_dir,
            description="Lists the contents of a directory inside the workspace sandbox.",
            input_map={
                "path": "path",
            },
            requires_permission=False
        ),
        "summarise_txt": make_tool_registry_entry(
            func=summarise_txt,
            description="Summarises text produced by a previous step.",
            input_map={
                "text": "source_step",
            },
            requires_permission=False
        ),
    }