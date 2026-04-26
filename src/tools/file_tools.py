'''
        All followins tools can only read files (or list directories) based on the respective folder which is workspace, 
        i.e. it will only be able to access files if they are on the same directory. Hence this will enforce workspace sandboxing so the respective system it runs on
        is untouched.
    '''
from src.tools.utils import WorkspaceConfig

def read_file(workspace: WorkspaceConfig, path: str) -> str:
    """
        Reads a text file safely from inside the workspace sandbox.
    """
    file_path = workspace.resolve_workspace_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    if not file_path.is_file():
        raise ValueError(f"'{path}' is not a file")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def list_dir(workspace: WorkspaceConfig, path: str = ".",) -> list[str]:
    """
        Lists the contents of a directory safely from inside the workspace sandbox.
        Further point i set the local variable path to "." as a default value so the system can 
        know about anything which is in the workspace folder too and not just subdirectories
    """
    dir_path = workspace.resolve_workspace_path(path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{path}' not found")

    if not dir_path.is_dir():
        raise ValueError(f"'{path}' is not a directory")

    return [item.name for item in dir_path.iterdir()]

def find_file(workspace: WorkspaceConfig, query: str, directory: str = ".") -> str:
    """
    Finds exactly one file inside the workspace matching the query.
    Returns:
        Exact relative file path as a string.
    Fails if:
        - no file matches
        - more than one file matches
    """
    dir_path = workspace.resolve_workspace_path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' not found")

    if not dir_path.is_dir():
        raise ValueError(f"'{directory}' is not a directory")

    matches = []

    for item in dir_path.iterdir():
        if item.is_file() and query.lower() in item.name.lower():
            matches.append(item.name)

    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching '{query}' found")

    if len(matches) > 1:
        raise ValueError(f"Multiple files match '{query}': {matches}")

    return matches[0]

def create_file(workspace: WorkspaceConfig, path: str, content: str = "") -> str:
    file_path = workspace.resolve_workspace_path(path)

    if file_path.exists():
        raise FileExistsError(f"File '{path}' already exists")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"File '{path}' created successfully."


def write_file(workspace: WorkspaceConfig, path: str, content: str) -> str:
    file_path = workspace.resolve_workspace_path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"File '{path}' written successfully."


def append_file(workspace: WorkspaceConfig, path: str, content: str) -> str:
    file_path = workspace.resolve_workspace_path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)

    return f"Content appended to '{path}' successfully."

def snapshot_file(workspace: WorkspaceConfig, path: str) -> dict:
    file_path = workspace.resolve_workspace_path(path)

    if file_path.exists():
        return {
            "path": path,
            "existed": True,
            "content": file_path.read_text(encoding="utf-8")
        }

    return {
        "path": path,
        "existed": False,
        "content": None
    }


def rollback_file_snapshot(workspace: WorkspaceConfig, snapshot: dict):
    file_path = workspace.resolve_workspace_path(snapshot["path"])

    if snapshot["existed"]:
        file_path.write_text(snapshot["content"], encoding="utf-8")
    else:
        if file_path.exists():
            file_path.unlink()