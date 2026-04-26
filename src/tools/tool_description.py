def make_tool_description(name: str, description: str, args_schema: dict[str, dict[str,str]], returns: dict) -> dict:
    """
        Creates a planner-facing tool description, so that the llm understands what tools exist and what arguments they require.
    """
    return {"name": name, "description": description, "args_schema": args_schema, "returns": returns}


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
            "Do not use this if the exact file path is unknown; use find_file when the user gives a partial filename, or list_dir only when the user asks to inspect available files. "
            "Use this before summarise_txt when the file must be read before it can be summarised. "
            "The 'path' argument must be the relative path of the file inside the workspace. "
            "Returns: the file contents as a string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace",
                "step_chainable": True
            }
        },
        returns= {"type": "string"}
    ),

    make_tool_description(
        name="list_dir",
        description=(
            "List files and subdirectories inside a directory in the workspace sandbox. "
            "Use this when the user wants to inspect available files, explore the workspace structure, or when the exact file path is not yet known. "
            "Do not use this to read file contents; use read_file for that. "
            "Use this before read_file if you first need to discover where a file is located. "
            "The 'path' argument must be a relative directory path inside the workspace, and '.' means the workspace root. "
            "Returns: JSON array of name strings, e.g. [\"main.py\", \"data/\", \"README.md\"]. "
            "Directories are suffixed with '/'. Use these names to construct paths for read_file."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative directory path inside workspace. Use '.' to list workspace root.",
                "step_chainable": False
            }
        },
        returns= {"type": "array"}
    ),

    make_tool_description(
        name="find_file",
        description=(
            "Find exactly one file inside the workspace sandbox using a partial filename query. "
            "Use this when the user clearly refers to a file but does not provide the exact filename or extension. "
            "Do not use this if the user already provided the exact filename. "
            "Use this before read_file when read_file needs an exact path but the user gave only a partial file name. "
            "Fails if no files match or if multiple files match."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Partial filename or search term to match.",
                "step_chainable": False
            },
            "directory": {
                "type": "string",
                "description": "Directory to search inside. Use '.' for the workspace root.",
                "step_chainable": False
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="summarise_txt",
        description=(
            "Summarise text produced by a previous step. "
            "Use this only when the user explicitly asks to summarise, summarize, shorten, explain, or describe text/file contents. "
            "Do not use this when the user only asks to read, show, open, or view a file. "
            "Use text_step when summarising output from a previous step such as read_file. "
            "Do not pass a file path directly."
        ),
        args_schema={
            "text":{
                "type": "string",
                "description": "Text to summarise. Use text_step if the text comes from a previous step.",
                "step_chainable": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="create_file",
        description=(
            "Create a new file inside the workspace sandbox. "
            "Use this when the user asks to create a new file. "
            "If the user also gives content to put inside the file, place that content in the content argument. "
            "Do not use this to overwrite an existing file. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the new file inside the workspace.",
                "step_chainable": False
            },
            "content": {
                "type": "string",
                "description": "Initial text content to write into the file. Can use content_step if content comes from a previous step.",
                "step_chainable": True
            }
        },
        returns={"type": "string"}
    ),
    
    make_tool_description(
        name="write_file",
        description=(
            "Write text content to a file inside the workspace sandbox. "
            "Use this when the user asks to write, save, or overwrite content in a file. "
            "Use content_step if the content comes from a previous step such as summarise_txt. "
            "Do not use this when the user only wants the result displayed. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
                "step_chainable": False
            },
            "content": {
                "type": "string",
                "description": "Text content to write. Can use content_step if content comes from a previous step.",
                "step_chainable": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="append_file",
        description=(
            "Append text content to the end of a file inside the workspace sandbox. "
            "Use this when the user asks to add content without replacing existing content. "
            "Use content_step if the content comes from a previous step. "
            "Do not use this to overwrite the whole file. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
                "step_chainable": False
            },
            "content": {
                "type": "string",
                "description": "Text content to append. Can use content_step if content comes from a previous step.",
                "step_chainable": True
            }
        },
        returns={"type": "string"}
    ),
    
    make_tool_description(
        name="direct_response",
        description=(
            "Generate a direct response to the user using the LLM. "
            "Use this for normal chat tasks such as greetings, explanations, email drafts, message drafts, rewrites, or questions. "
            "Do not use this after read_file just to display file contents. The Coordinator/runner displays read_file results directly. "
            "Do not use this after summarise_txt just to display the summary. The Coordinator/runner displays summarise_txt results directly. "
            "Do not use this if the user explicitly asks to create, write, append, or overwrite a file."
        ),
        args_schema={
            "prompt": {
                "type": "string",
                "description": "The instruction or user request that the LLM should answer directly.",
                "step_chainable": True
            }
        },
        returns={"type": "string"}
    )
]