from typing import Any, Callable

from src.OllamaClient import OllamaClient
from src.tools.file_tools import read_file, list_dir, write_file
from src.tools.llm_tools import create_summarise_txt_func


def make_tool_registry_entry(func: Callable[..., Any], description: str, input_map: dict[str, str]) -> dict[str, Any]:
    """
        Creates an executor-facing tool registry entry.
        - func: actual callable Python function
        - description: human-readable explanation
        - input_map: maps real function arguments -> planner argument names
    """
    return {"func": func, "description": description, "input_map": input_map}


def build_tool_registry(llm_client: OllamaClient) -> dict[str, dict[str, Any]]:
    """
        Builds the executor-facing tool registry.
        The summarisation tool is bound to the provided llm_client here.
    """
    summarise_txt = create_summarise_txt_func(llm_client)

    return {
        "read_file": make_tool_registry_entry(
            func=read_file,
            description="Reads the contents of a text file from inside the workspace sandbox.",
            input_map={
                "file_id": "file_id",
            },
        ),
        "list_dir": make_tool_registry_entry(
            func=list_dir,
            description="Lists the contents of a directory inside the workspace sandbox.",
            input_map={
                "path": "path",
            },
        ),
        "summarise_txt": make_tool_registry_entry(
            func=summarise_txt,
            description="Summarises text produced by a previous step.",
            input_map={
                "text": "source_step",
            },
        ),
        "write_file": make_tool_registry_entry(
            func=write_file,
            description="Writes text content into a file inside the workspace sandbox.",
            input_map={
                "filename": "filename",
                "content": "source_step",
            },
        ),
    }