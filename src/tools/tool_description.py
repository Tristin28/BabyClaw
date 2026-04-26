def make_tool_description(name: str, description: str, args_schema: dict[str, dict[str, object]], returns: dict) -> dict:
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
            "Do not use this if the exact file path is unknown; use find_file or find_file_recursive when the user gives a partial filename. "
            "Use this before summarise_txt when the file must be read before it can be summarised. "
            "The 'path' argument must be the relative path of the file inside the workspace. "
            "Returns: the file contents as a string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="list_dir",
        description=(
            "List files and subdirectories directly inside one directory in the workspace sandbox. "
            "Use this when the user wants to inspect one folder level only. "
            "Do not use this to recursively inspect subdirectories; use list_tree for that. "
            "Do not use this to read file contents; use read_file for that. "
            "The 'path' argument must be a relative directory path inside the workspace, and '.' means the workspace root. "
            "Returns: an array of name strings."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative directory path inside workspace. Use '.' to list workspace root.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "array"}
    ),

    make_tool_description(
        name="find_file",
        description=(
            "Find exactly one file inside one directory in the workspace sandbox using a partial filename query. "
            "Use this when the user clearly refers to a file but does not provide the exact filename or extension, and the file is expected in one directory. "
            "Do not use this if the user already provided the exact filename. "
            "Do not use this if the file may be inside subdirectories; use find_file_recursive for that. "
            "Use this before read_file when read_file needs an exact path but the user gave only a partial file name. "
            "Fails if no files match or if multiple files match."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Partial filename or search term to match.",
                "step_chainable": False,
                "required": True
            },
            "directory": {
                "type": "string",
                "description": "Directory to search inside. Use '.' for the workspace root.",
                "step_chainable": False,
                "required": True
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
            "text": {
                "type": "string",
                "description": "Text to summarise. Use text_step if the text comes from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="create_file",
        description=(
            "Create a new file inside the workspace sandbox. "
            "Use this when the user asks to create a new file. "
            "If the user gives content to put inside the file, place that content in the content argument. "
            "If the user asks for an empty file, use content as an empty string. "
            "Never call create_file with only path; create_file always needs both path and content. "
            "Do not use this to overwrite an existing file. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the new file inside the workspace.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Initial text content to write into the file. Use content_step if content comes from a previous step. Use an empty string if the user wants an empty file.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="write_file",
        description=(
            "Write text content to a file inside the workspace sandbox. "
            "Use this when the user asks to write, save, replace, or overwrite content in a file. "
            "Use content_step if the content comes from a previous step such as direct_response, read_file, or summarise_txt. "
            "Do not use this when the user only wants the result displayed. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Text content to write. Use content_step if content comes from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="append_file",
        description=(
            "Append text content to the end of a file inside the workspace sandbox. "
            "Use this when the user asks to add content without replacing existing content. "
            "If the user directly gives the text to append, use the content argument directly. "
            "Use content_step only if the content comes from a previous step. "
            "Do not use this to overwrite the whole file. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Text content to append. Use content_step if content comes from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="direct_response",
        description=(
            "Generate a direct response to the user using the LLM. "
            "Use this for normal chat tasks such as greetings, explanations, email drafts, message drafts, rewrites, or questions. "
            "Use this when the user asks to generate text but does not ask to save it into a file. "
            "Do not use this after read_file just to display file contents. The Coordinator/runner displays read_file results directly. "
            "Do not use this after summarise_txt just to display the summary. The Coordinator/runner displays summarise_txt results directly. "
            "Do not use this if the user explicitly asks to create, write, append, overwrite, delete, move, or copy a file/folder."
        ),
        args_schema={
            "prompt": {
                "type": "string",
                "description": "The instruction or user request that the LLM should answer directly.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="delete_file",
        description=(
            "Delete a file inside the workspace sandbox. "
            "Use this only when the user explicitly asks to delete or remove a file. "
            "Do not use this to delete folders; use delete_dir for folders. "
            "Do not use this for clearing file content; use write_file with empty content only if the user asks to empty a file. "
            "Requires permission because it modifies the workspace. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to delete inside the workspace.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="search_text",
        description=(
            "Search for a text query inside files in the workspace sandbox. "
            "Use this when the user asks to find where some word, phrase, function, class, or concept appears inside workspace files. "
            "Do not use this to read a whole file; use read_file for that. "
            "Do not use this to find a filename; use find_file or find_file_recursive for filename search. "
            "Returns a list of matches with path, line number, and line text."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "The word or phrase to search for inside file contents.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "array"}
    ),

    make_tool_description(
        name="replace_text",
        description=(
            "Replace text inside a file in the workspace sandbox. "
            "Use this when the user asks to replace one exact piece of text with another exact piece of text inside a file. "
            "Do not use this if the user asks to rewrite the whole file; use write_file for that. "
            "If the file path is unknown, use find_file or find_file_recursive first, then use path_step. "
            "Requires permission because it modifies the file. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to edit.",
                "step_chainable": True,
                "required": True
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace.",
                "step_chainable": False,
                "required": True
            },
            "new_text": {
                "type": "string",
                "description": "New text to insert instead.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="list_tree",
        description=(
            "Recursively list files and folders inside a workspace directory, including subdirectories. "
            "Use this when the user asks to look through folders, inspect the project tree, show all files, or explore subdirectories. "
            "Use path='.' for the workspace root. "
            "Do not use this to read file contents. Use read_file for contents. "
            "Returns a list of relative paths. Directories end with '/'."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative directory path to recursively list. Use '.' for workspace root.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "array"}
    ),

    make_tool_description(
        name="find_file_recursive",
        description=(
            "Recursively find exactly one file by partial filename inside a directory and its subdirectories. "
            "Use this when the user refers to a file but does not provide its exact path, and the file may be inside subdirectories. "
            "Do not use this when the user already gave the exact path. "
            "Use this before read_file, copy_file, copy_path, move_path, or replace_text when an exact path is needed. "
            "Fails if no files match or if multiple files match."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Partial filename to search for.",
                "step_chainable": False,
                "required": True
            },
            "directory": {
                "type": "string",
                "description": "Directory to search inside. Use '.' for workspace root.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="create_dir",
        description=(
            "Create a directory inside the workspace sandbox. "
            "Use this when the user asks to create a folder or directory. "
            "Do not use create_file for folders. "
            "Do not use this before create_file only to create parent folders, because create_file already creates parent folders when needed. "
            "Use this for empty folders or when the user explicitly asks for a folder. "
            "Requires permission because it modifies the workspace. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the directory to create.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="delete_dir",
        description=(
            "Delete a directory and everything inside it from the workspace sandbox. "
            "Use this only when the user explicitly asks to delete/remove a folder or directory. "
            "Do not use this to delete a file. Use delete_file for files. "
            "Requires permission because it modifies the workspace. "
            "Returns a confirmation string."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the directory to delete.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="move_path",
        description=(
            "Move a file or directory from one workspace path to another. "
            "Use this when the user asks to move or rename a file/folder. "
            "The destination_path must be the full final path, including the filename or folder name. "
            "Example: to move hello.txt into archive, use destination_path='archive/hello.txt'. "
            "Example: to rename old.txt to new.txt, use source_path='old.txt' and destination_path='new.txt'. "
            "Requires permission because it modifies the workspace. "
            "Returns a confirmation string."
        ),
        args_schema={
            "source_path": {
                "type": "string",
                "description": "Relative path of the source file or directory.",
                "step_chainable": True,
                "required": True
            },
            "destination_path": {
                "type": "string",
                "description": "Relative final destination path.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="copy_path",
        description=(
            "Copy a file or directory from one workspace path to another. "
            "Use this when the user asks to copy, duplicate, or clone a file/folder. "
            "The destination_path must be the full final path, including the filename or folder name. "
            "Example: to copy hello.txt into backup, use destination_path='backup/hello.txt'. "
            "Use this instead of copy_file if the source may be either a file or a folder. "
            "Requires permission because it writes to the workspace. "
            "Returns a confirmation string."
        ),
        args_schema={
            "source_path": {
                "type": "string",
                "description": "Relative path of the source file or directory.",
                "step_chainable": True,
                "required": True
            },
            "destination_path": {
                "type": "string",
                "description": "Relative final destination path.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),
]