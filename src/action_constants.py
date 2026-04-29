'''
    These global variables are representing the system’s rulebook for what actions the agents are allowed to use.
    Where each component (module) share the same set of rules hence why these variables are centralised here to ensure consistency and ease of maintenance.
'''
DIRECT_RESPONSE_TOOLS = {"direct_response"}

READ_FILE_TOOLS = {
    "read_file",
    "find_file",
    "find_file_recursive",
    "list_dir",
    "list_tree",
    "search_text",
}

SUMMARISE_FILE_TOOLS = {
    "read_file",
    "find_file",
    "find_file_recursive",
    "summarise_txt",
}

MUTATION_TOOLS = {
    "create_file",
    "write_file",
    "append_file",
    "delete_file",
    "replace_text",
    "create_dir",
    "delete_dir",
    "move_path",
    "copy_path",
}

CONTENT_WRITING_TOOLS = {
    "create_file",
    "write_file",
    "append_file",
}

CONTENT_MUTATION_TOOLS = {
    "write_file",
    "append_file",
    "replace_text",
}

MUTATION_PATH_ARGS = {
    "path",
    "source_path",
    "destination_path",
    "directory",
}

MUTATION_FILE_TOOLS = {
    "generate_content",
    "create_file",
    "write_file",
    "append_file",
    "delete_file",
    "replace_text",
    "create_dir",
    "delete_dir",
    "move_path",
    "copy_path",
    "find_file",
    "find_file_recursive",
    "list_dir",
    "list_tree",
    "read_file",
    "summarise_txt",
    "search_text",
}

GENERATED_ARTIFACT_TERMS = {
    "program",
    "app",
    "application",
    "game",
    "script",
    "pipeline",
    "algorithm",
    "system",
}

CREATIVE_ARTIFACT_PREFIXES = (
    "a poem",
    "an poem",
    "poem",
    "a story",
    "short story",
    "a letter",
    "an email",
)
