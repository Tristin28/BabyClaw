def make_tool_description(name: str, description: str, args_schema: dict[str, dict[str, object]], returns: dict) -> dict:
    """
        Creates a planner-facing tool description.
        These descriptions are shown to the planning LLM, so they should explain
        when to choose each tool and how to combine tools safely.
    """
    return {"name": name, "description": description, "args_schema": args_schema, "returns": returns}


PLANNER_TOOL_DESCRIPTIONS = [
    make_tool_description(
        name="direct_response",
        description=(
            "Produce the final answer directly in chat without reading or changing workspace files. "
            "Use this for normal conversation, explanations, advice, drafting text, rewriting text, answering questions, and writing code that the user wants displayed in chat. "
            "Use this when the user asks to generate content but does not ask to save it into a file. "
            "Do not use this when the user explicitly asks to create, save, write, append, edit, replace, delete, move, copy, rename, list, read, search, or summarise workspace files/folders. "
            "Do not add direct_response after read_file, summarise_txt, list_dir, list_tree, find_file, or search_text just to display their result; the coordinator displays tool results. "
            "The prompt should restate the user's actual request faithfully and preserve pronouns."
        ),
        args_schema={
            "prompt": {
                "type": "string",
                "description": "The exact instruction/question to answer in chat. Preserve the user's meaning, topic, constraints, and pronouns.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="generate_content",
        description=(
            "Generate raw text/code/config/document content for a later file-writing step. "
            "Use this only when the user asks to create/save/write a file whose content must be generated rather than directly provided by the user. "
            "Always chain this into create_file or write_file using content_step. A generate_content-only plan does not complete a workspace creation task. "
            "Do not use this for normal chat replies; use direct_response. "
            "Do not use this when the user already provided the exact content to write; pass that content directly to create_file/write_file/append_file. "
            "The prompt must be specific enough to satisfy the requested artifact: include language, format, topic, required behavior, and output-only instruction when relevant. "
            "For programs/apps/games/scripts/pipelines, ask for a meaningful minimal implementation, not a placeholder or future plan."
        ),
        args_schema={
            "prompt": {
                "type": "string",
                "description": "Specific generation instruction for the file content. Mention artifact type, language/format, topic, required behavior, and that only raw content should be returned.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="read_file",
        description=(
            "Read the full text contents of one existing file inside the workspace. "
            "Use this when the user asks to read, show, open, view, inspect, or use the contents of a specific file. "
            "Use this before summarise_txt when the user asks to summarise/explain/shorten a file. "
            "If the user gives an exact relative path, use it directly. "
            "If the file name is partial or unclear, use find_file or find_file_recursive first, then read_file with path_step. "
            "Do not use this to list folders; use list_dir or list_tree. "
            "Do not use this to search for text across many files; use search_text. "
            "Do not add direct_response after this just to show the file content."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the workspace. Use path_step only when an earlier find tool produced the path.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="summarise_txt",
        description=(
            "Summarise, explain, shorten, or describe text that has already been produced by an earlier step. "
            "Use this for file summaries by first reading the file, then calling summarise_txt with text_step. "
            "Typical file-summary plan: read_file -> summarise_txt(text_step=1). "
            "Do not pass a file path as text. If the user names a file, read it first. "
            "Do not use this when the user only asks to read/show/view the file; read_file alone is enough. "
            "Do not add direct_response after this just to display the summary."
        ),
        args_schema={
            "text": {
                "type": "string",
                "description": "Text to summarise. Usually use text_step from read_file or another text-producing step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="list_dir",
        description=(
            "List the immediate files/folders directly inside one workspace directory. "
            "Use this when the user asks to list a folder, show files in a directory, or inspect one directory level. "
            "Use path='.' for the workspace root. "
            "Use list_tree instead when the user asks recursively, asks for the project tree, or wants nested files. "
            "Do not use this to read file contents; use read_file. "
            "Do not use this before create_file just to check whether parent folders exist."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative directory path. Use '.' for the workspace root.",
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
            "Use this when the user asks for the project tree, all files, recursive listing, nested structure, or a broader workspace overview. "
            "Use path='.' for the workspace root. "
            "Use list_dir instead when only one directory level is requested. "
            "Do not use this to read file contents; use read_file. "
            "Do not add direct_response after this just to display the listing."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative directory path to recursively list. Use '.' for the workspace root.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "array"}
    ),

    make_tool_description(
        name="find_file",
        description=(
            "Find exactly one file by partial filename inside one known directory. "
            "Use this when the user refers to a file unclearly and it should be in a specific directory. "
            "Use find_file_recursive instead if the file may be nested in subdirectories. "
            "Do not use this when the user already provided an exact path. "
            "Do not use this to search inside file contents; use search_text. "
            "Common chains: find_file -> read_file(path_step=1), find_file -> replace_text(path_step=1), find_file -> delete_file(path_step=1). "
            "If multiple files may match, prefer find_file_recursive only when nesting is the issue; otherwise the tool should fail rather than guessing."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Partial filename or filename clue from the user. Do not invent unrelated names.",
                "step_chainable": False,
                "required": True
            },
            "directory": {
                "type": "string",
                "description": "Directory to search inside. Use '.' for workspace root unless the user specified a directory.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="find_file_recursive",
        description=(
            "Find exactly one file by partial filename inside a directory and all nested subdirectories. "
            "Use this when the user refers to a file unclearly and it may be anywhere under a directory. "
            "Do not use this when the user already provided an exact path. "
            "Do not use this to search inside file contents; use search_text. "
            "Common chains: find_file_recursive -> read_file(path_step=1), find_file_recursive -> replace_text(path_step=1), find_file_recursive -> move_path(source_path_step=1). "
            "Use directory='.' unless the user specified a narrower directory."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Partial filename or filename clue from the user. Do not invent unrelated names.",
                "step_chainable": False,
                "required": True
            },
            "directory": {
                "type": "string",
                "description": "Directory to search from recursively. Use '.' for workspace root unless the user specified a directory.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="search_text",
        description=(
            "Search for a word, phrase, function, class, concept, or code fragment inside workspace file contents. "
            "Use this when the user asks where something appears, asks to find text in files, or asks to locate references/usages. "
            "Do not use this to find filenames; use find_file or find_file_recursive. "
            "Do not use this to read an entire file; use read_file after the user asks for a specific file or after search results identify the target. "
            "Do not add direct_response after this just to display search results."
        ),
        args_schema={
            "query": {
                "type": "string",
                "description": "Exact word, phrase, symbol, or code fragment to search for inside files.",
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
            "Use this when the user asks to create a new file or save a newly generated artifact to a new file. "
            "Always provide path and content. Use content='' only when the user asks for an empty file or gives no content requirement. "
            "If the user directly provides the content, pass it directly as content. "
            "If the user asks for generated content, first use generate_content, then create_file with content_step. "
            "Do not use this to overwrite an existing file; use write_file. "
            "Do not invent filenames or extensions. Preserve the path/name the user gave. If no filename/path is given, the plan should not invent generic names like output.txt. "
            "Do not create extra folders first unless the user explicitly asked for an empty folder; create_file can create parent directories when needed."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the new file. Preserve the user's exact filename/path and extension.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Initial file content. Use content_step when generated or read from a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="write_file",
        description=(
            "Write or overwrite the complete content of one file inside the workspace. "
            "Use this when the user asks to write to a file, save content to a file, overwrite a file, replace the whole file, or create/update a file where overwriting is acceptable. "
            "If the user directly provides the content, pass it directly as content. "
            "If the content must be generated, first use generate_content, then write_file with content_step. "
            "Do not use this for chat-only answers; use direct_response. "
            "Do not use this for appending; use append_file. "
            "Do not use this for exact small replacements inside existing text; use replace_text. "
            "Preserve the user's requested path/name and do not invent generic filenames."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to write/overwrite. Preserve the user's exact path.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Complete text to write. Use content_step when generated or produced by a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="append_file",
        description=(
            "Append text to the end of an existing or new file without replacing existing content. "
            "Use this when the user says append, add to the end, add below, continue the file, or preserve existing content while adding new content. "
            "If the user directly provides the appended text, pass it directly as content. "
            "If the appended content must be generated, first use generate_content, then append_file with content_step. "
            "Do not use this to overwrite or replace the whole file; use write_file. "
            "Do not use this for exact replacement of existing text; use replace_text. "
            "Preserve the user's requested path."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to append to. Preserve the user's exact path.",
                "step_chainable": False,
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Text to append. Use content_step when generated or produced by a previous step.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="replace_text",
        description=(
            "Replace exact text inside one existing file. "
            "Use this when the user gives old text and new text, or clearly asks to replace a specific phrase/block with another phrase/block. "
            "If the path is exact, use it directly. If the path is unclear, use find_file or find_file_recursive first, then replace_text with path_step. "
            "Do not use this to rewrite an entire file; use write_file. "
            "Do not use this when the user did not specify what old text should be replaced. "
            "old_text must be the exact text to find. new_text must be the exact replacement text."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to edit. Use path_step only when an earlier find tool produced the path.",
                "step_chainable": True,
                "required": True
            },
            "old_text": {
                "type": "string",
                "description": "Exact existing text to replace. Do not invent it if the user did not provide it.",
                "step_chainable": False,
                "required": True
            },
            "new_text": {
                "type": "string",
                "description": "Exact new text to insert in place of old_text.",
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
            "If the file path is exact, use it directly. If unclear, use find_file or find_file_recursive first, then delete_file with path_step. "
            "Do not use this for directories; use delete_dir. "
            "Do not use this to clear file content; use write_file with empty content only if the user explicitly asks to empty the file. "
            "Never delete files that the user did not mention."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the file to delete. Use path_step only when an earlier find tool produced the path.",
                "step_chainable": True,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="create_dir",
        description=(
            "Create one directory/folder inside the workspace. "
            "Use this when the user explicitly asks to create a folder/directory, especially an empty folder or project folder. "
            "Do not use this before create_file/write_file just to create parent folders; file-writing tools can create needed parent folders. "
            "Do not invent folder names. Preserve the user's requested directory path. "
            "If the user asks to create a project with files, use create_dir only if the folder itself is explicitly requested, then create/write the requested files."
        ),
        args_schema={
            "path": {
                "type": "string",
                "description": "Relative path of the directory to create. Preserve the user's exact folder name/path.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="delete_dir",
        description=(
            "Delete one directory/folder and everything inside it from the workspace. "
            "Use this only when the user explicitly asks to delete/remove a directory/folder. "
            "If the directory path is exact, use it directly. If unclear, use find/list tools only when needed to resolve the requested directory. "
            "Do not use this for files; use delete_file. "
            "Never delete directories that the user did not mention."
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
            "Move or rename one file/folder inside the workspace. "
            "Use this when the user asks to move, rename, relocate, or change the path/name of a file or folder. "
            "source_path is the current path. destination_path is the full final path, including the final filename/folder name. "
            "For rename: source_path='old.txt', destination_path='new.txt'. "
            "For move: source_path='notes.txt', destination_path='archive/notes.txt'. "
            "If the source path is unclear, use find_file or find_file_recursive first, then move_path with source_path_step. "
            "Do not use copy_path when the original should disappear. Do not move extra paths."
        ),
        args_schema={
            "source_path": {
                "type": "string",
                "description": "Relative source path. Use source_path_step only when an earlier find tool produced the source path.",
                "step_chainable": True,
                "required": True
            },
            "destination_path": {
                "type": "string",
                "description": "Relative full final destination path, including filename/folder name.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),

    make_tool_description(
        name="copy_path",
        description=(
            "Copy one file/folder inside the workspace while keeping the original. "
            "Use this when the user asks to copy, duplicate, clone, or make a backup of a file/folder. "
            "source_path is the original path. destination_path is the full copied path, including the final filename/folder name. "
            "Example: source_path='notes.txt', destination_path='backup/notes.txt'. "
            "If the source path is unclear, use find_file or find_file_recursive first, then copy_path with source_path_step. "
            "Do not use move_path when the original should remain. Do not copy extra paths."
        ),
        args_schema={
            "source_path": {
                "type": "string",
                "description": "Relative source path. Use source_path_step only when an earlier find tool produced the source path.",
                "step_chainable": True,
                "required": True
            },
            "destination_path": {
                "type": "string",
                "description": "Relative full final destination path, including filename/folder name.",
                "step_chainable": False,
                "required": True
            }
        },
        returns={"type": "string"}
    ),
]
