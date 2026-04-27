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
        name="direct_response",
        description=(
            "Answer the user directly without using workspace files. "
            "Use this for normal conversation, explanations, advice, drafting, rewriting, and questions. "
            "Use this when the user wants generated text shown in chat, not saved to a file. "
            "Do not use this after read_file or summarise_txt just to display the result. "
            "Do not use this when the user explicitly asks to create, write, append, delete, move, copy, or edit files/folders."
        ),
        args_schema={
            "prompt": {
                "type": "string",
                "description": "The exact instruction/question to answer. Preserve the user's meaning and pronouns.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="generate_content",
        description=(
            "Generate raw text/code content that will be saved into a file by a later step. "
            "Use this only when the user asks to generate something and save/create/write it into a file. "
            "Typical order: generate_content first, then create_file or write_file with content_step. "
            "Return only the raw content, with no explanation, greeting, or markdown code fences. "
            "Do not use this for normal chat replies; use direct_response instead."
        ),
        args_schema={
            "prompt": {
                "type": "string",
                "description": "Specific instruction describing the content to generate. Mention format/language when relevant.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="read_file",
        description=(
            "Read the full contents of one file inside the workspace. "
            "Use this when the user asks to read, show, open, view, or use the contents of an exact file path. "
            "Use this before summarise_txt when summarising a file. "
            "If the file path is unclear, use find_file or find_file_recursive first, then read_file with path_step. "
            "Do not use this to list folders or search filenames."
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
        name="summarise_txt",
        description=(
            "Summarise text from a previous step. "
            "Use this when the user asks to summarise, shorten, explain, or describe file/text contents. "
            "Typical order for a file summary: read_file first, then summarise_txt with text_step. "
            "Do not pass a file path directly. "
            "Do not use this when the user only asks to read/show/open/view a file."
        ),
        args_schema={
            "text": {
                "type": "string",
                "description": "Text to summarise. Usually use text_step from read_file.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="list_dir",
        description=(
            "List the files/folders directly inside one workspace directory. "
            "Use this when the user asks to list one folder level. "
            "Use path='.' for the workspace root. "
            "Do not use this to recursively inspect subfolders; use list_tree. "
            "Do not use this to read file contents; use read_file."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative directory path. Use '.' for workspace root.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "array"}
    ),

    make_tool_description(
        name="list_tree",
        description=(
            "Recursively list files/folders inside a workspace directory. "
            "Use this when the user asks to inspect the project tree, show all files, or look inside subdirectories. "
            "Use path='.' for the workspace root. "
            "Do not use this to read contents; use read_file."
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
        name="find_file",
        description=(
            "Find exactly one file by partial filename inside one directory. "
            "Use this when the user refers to a file unclearly and it is expected in one known directory. "
            "Use find_file_recursive if it may be inside subdirectories. "
            "Do not use this if the user already gave the exact path. "
            "Typical order: find_file, then read_file/replace_text/move_path/copy_path using path_step."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Partial filename or search term.",
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
        name="find_file_recursive",
        description=(
            "Find exactly one file by partial filename inside a directory and all its subdirectories. "
            "Use this when the user refers to a file unclearly and it may be nested. "
            "Do not use this if the user already gave the exact path. "
            "Typical order: find_file_recursive, then read_file/replace_text/move_path/copy_path using path_step."
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
                "description": "Directory to search from. Use '.' for workspace root.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="search_text",
        description=(
            "Search for a word/phrase inside workspace file contents. "
            "Use this when the user asks where some text, function, class, phrase, or concept appears. "
            "Do not use this to find filenames; use find_file or find_file_recursive. "
            "Do not use this to read a whole file; use read_file."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Text to search for inside files.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "array"}
    ),

    make_tool_description(
        name="create_file",
        description=(
            "Create a brand-new file inside the workspace. "
            "Use this only when the user asks to create a new file. "
            "Always provide both path and content. "
            "If the user gives direct content, use it directly. "
            "If the user asks for an empty file, use content=''. "
            "If content must be generated, use generate_content first and then create_file with content_step. "
            "Do not use this to overwrite an existing file; use write_file."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the new file. Preserve the user's filename/path.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Initial file content. Use content_step if generated/read from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="write_file",
        description=(
            "Write/overwrite text content into a file inside the workspace. "
            "Use this when the user asks to write, save, replace the whole file, or overwrite a file. "
            "If the user gives direct content, use it directly. "
            "If content must be generated, use generate_content first and then write_file with content_step. "
            "Do not use this when the user only wants to see the answer in chat."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to write.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Text to write. Use content_step if it comes from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="append_file",
        description=(
            "Append text to the end of a file inside the workspace. "
            "Use this when the user asks to add content without replacing existing content. "
            "If the user gives direct content, use it directly. "
            "Use content_step only if the content comes from a previous step. "
            "Do not use this to overwrite the whole file."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to append to.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Text to append. Use content_step if it comes from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="replace_text",
        description=(
            "Replace exact text inside one file. "
            "Use this when the user asks to replace one exact piece of text with another. "
            "If the file path is unknown, use find_file or find_file_recursive first, then replace_text with path_step. "
            "Do not use this to rewrite the whole file; use write_file."
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
                "description": "New text to insert.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="delete_file",
        description=(
            "Delete one file inside the workspace. "
            "Use this only when the user explicitly asks to delete/remove a file. "
            "Do not use this for folders; use delete_dir. "
            "Do not use this to clear file content; use write_file with empty content only if the user asks to empty the file."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to delete.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="create_dir",
        description=(
            "Create a folder/directory inside the workspace. "
            "Use this when the user explicitly asks to create a folder/directory. "
            "Do not use this before create_file just to create parent folders, because create_file can create parent folders when needed. "
            "Use this for empty folders or explicitly requested folders."
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
            "Delete one directory and its contents from the workspace. "
            "Use this only when the user explicitly asks to delete/remove a folder/directory. "
            "Do not use this for files; use delete_file."
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
            "Move or rename a file/folder inside the workspace. "
            "Use this when the user asks to move or rename something. "
            "destination_path must be the full final path, including the filename/folder name. "
            "Example move: source_path='hello.txt', destination_path='archive/hello.txt'. "
            "Example rename: source_path='old.txt', destination_path='new.txt'."
        ),
        args_schema={
            "source_path": {
                "type": "string",
                "description": "Relative source path.",
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
            "Copy a file/folder inside the workspace. "
            "Use this when the user asks to copy, duplicate, or clone something. "
            "destination_path must be the full final path, including the filename/folder name. "
            "Example: source_path='hello.txt', destination_path='backup/hello.txt'."
        ),
        args_schema={
            "source_path": {
                "type": "string",
                "description": "Relative source path.",
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