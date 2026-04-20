def make_tool_description(name: str, description: str, args_schema: dict[str, str]) -> dict:
    """
        Creates a planner-facing tool description, so that the llm understands what tools exist and what arguments they require.
    """
    return {"name": name, "description": description, "args_schema": args_schema}


'''
    Description of a tool is a key feature because the model will decide what tool to pick based almost entirely on this field
    Hence why i followed a pipeline in order to describe it well and structured which explains these following things about a tool: 
    what it does, when to use it, when not to use it, dependency guidance, argument meaning and output meaning.
'''
PLANNER_TOOL_DESCRIPTIONS = [
    make_tool_description(
        name="read_file",
        description=(
            "Read the full contents of a file inside the workspace sandbox. "
            "Use this when the user asks about a specific file or when another step needs the file's text as input. "
            "Do not use this if the file location is unknown; use list_dir first to locate files or inspect directories. "
            "Use this before summarise_txt when the file must be read before it can be summarised. "
            "The 'file_id' argument must be the relative path of the file inside the workspace. "
            "This tool returns the file contents as a string."
        ),
        args_schema={
            "file_id": "string",
        },
    ),
    make_tool_description(
        name="list_dir",
        description=(
            "List files and subdirectories inside a directory in the workspace sandbox. "
            "Use this when the user wants to inspect available files, explore the workspace structure, or when the exact file path is not yet known. "
            "Do not use this to read file contents; use read_file for that. "
            "Use this before read_file if you first need to discover where a file is located. "
            "The 'path' argument must be a relative directory path inside the workspace, and '.' means the workspace root. "
            "This tool returns a list of file and subdirectory names."
        ),
        args_schema={
            "path": "string",
        },
    ),
    make_tool_description(
        name="summarise_txt",
        description=(
            "Summarise text produced by a previous step. "
            "Use this when another step has already returned text that needs to be shortened into a clear summary. "
            "Do not use this directly on a file path or directory path; use read_file first to obtain the text content. "
            "Use this after read_file or after another text-producing tool. "
            "The 'source_step' argument must be the id of the earlier step that produced the text to summarise. "
            "This tool returns a summary as a string."
        ),
        args_schema={
            "source_step": "integer",
        },
    )
]