def make_tool_description(name: str, description: str, args_schema: dict[str, str]) -> dict:
    """
        Creates a planner-facing tool description, so that the llm understands what tools exist and what arguments they require.
    """
    return {"name": name, "description": description, "args_schema": args_schema}

PLANNER_TOOL_DESCRIPTIONS = [
    make_tool_description(
        name="read_file",
        description="Reads the contents of a text file from inside the workspace sandbox.",
        args_schema={
            "file_id": "string",
        },
    ),
    make_tool_description(
        name="list_dir",
        description="Lists the contents of a directory inside the workspace sandbox.",
        args_schema={
            "path": "string",
        },
    ),
    make_tool_description(
        name="summarise_txt",
        description="Summarises text produced by a previous step.",
        args_schema={
            "source_step": "integer",
        },
    )
]