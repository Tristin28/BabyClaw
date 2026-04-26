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


def delete_file(workspace: WorkspaceConfig, path: str) -> str:
    """
        Deletes a file inside the workspace sandbox.
    """
    file_path = workspace.resolve_workspace_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    if not file_path.is_file():
        raise ValueError(f"'{path}' is not a file")

    file_path.unlink()

    return f"File '{path}' deleted successfully."

def search_text(workspace: WorkspaceConfig, query: str) -> list[dict]:
    """
        Searches for text inside all readable files in the workspace.
        Returns a list of matches.
    """
    root_path = workspace.resolve_workspace_path(".")
    matches = []

    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(lines, start=1):
            if query.lower() in line.lower():
                matches.append({
                    "path": str(file_path.relative_to(root_path)),
                    "line": line_number,
                    "text": line
                })

    return matches


def replace_text(workspace: WorkspaceConfig, path: str, old_text: str, new_text: str) -> str:
    """
        Replaces all occurrences of old_text with new_text inside a file.
    """
    file_path = workspace.resolve_workspace_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    if not file_path.is_file():
        raise ValueError(f"'{path}' is not a file")

    content = file_path.read_text(encoding="utf-8")

    if old_text not in content:
        raise ValueError(f"Text '{old_text}' was not found in '{path}'")

    updated_content = content.replace(old_text, new_text)

    file_path.write_text(updated_content, encoding="utf-8")

    return f"Text replaced successfully in '{path}'."

import shutil
from src.tools.utils import WorkspaceConfig


def list_tree(workspace: WorkspaceConfig, path: str = ".", max_depth: int = 5) -> list[str]:
    """
        Recursively lists files and folders inside a directory.
        Directories are shown with a trailing '/'.
    """
    root_path = workspace.resolve_workspace_path(path)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory '{path}' not found")

    if not root_path.is_dir():
        raise ValueError(f"'{path}' is not a directory")

    results = []

    for item in sorted(root_path.rglob("*")):
        relative_path = item.relative_to(root_path)
        depth = len(relative_path.parts)

        if depth > max_depth:
            continue

        item_name = str(relative_path)

        if item.is_dir():
            item_name += "/"

        results.append(item_name)

    return results


def find_file_recursive(workspace: WorkspaceConfig, query: str, directory: str = ".") -> str:
    """
        Recursively finds exactly one file inside a directory or its subdirectories.
    """
    root_path = workspace.resolve_workspace_path(directory)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' not found")

    if not root_path.is_dir():
        raise ValueError(f"'{directory}' is not a directory")

    matches = []

    for item in root_path.rglob("*"):
        if item.is_file() and query.lower() in item.name.lower():
            matches.append(str(item.relative_to(root_path)))

    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching '{query}' found inside '{directory}'")

    if len(matches) > 1:
        raise ValueError(f"Multiple files match '{query}': {matches}")

    return matches[0]


def create_dir(workspace: WorkspaceConfig, path: str) -> str:
    """
        Creates a directory inside the workspace sandbox.
    """
    dir_path = workspace.resolve_workspace_path(path)

    if dir_path.exists():
        raise FileExistsError(f"Directory or file '{path}' already exists")

    dir_path.mkdir(parents=True, exist_ok=False)

    return f"Directory '{path}' created successfully."


def delete_dir(workspace: WorkspaceConfig, path: str) -> str:
    """
        Deletes a directory and everything inside it.
    """
    dir_path = workspace.resolve_workspace_path(path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{path}' not found")

    if not dir_path.is_dir():
        raise ValueError(f"'{path}' is not a directory")

    shutil.rmtree(dir_path)

    return f"Directory '{path}' deleted successfully."


def move_path(workspace: WorkspaceConfig, source_path: str, destination_path: str) -> str:
    """
        Moves a file or directory inside the workspace sandbox.
        The destination path must be the full final path.
    """
    source = workspace.resolve_workspace_path(source_path)
    destination = workspace.resolve_workspace_path(destination_path)

    if not source.exists():
        raise FileNotFoundError(f"Source path '{source_path}' not found")

    if destination.exists():
        raise FileExistsError(f"Destination path '{destination_path}' already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(source), str(destination))

    return f"Moved '{source_path}' to '{destination_path}' successfully."


def copy_path(workspace: WorkspaceConfig, source_path: str, destination_path: str) -> str:
    """
        Copies a file or directory inside the workspace sandbox.
        The destination path must not already exist.
    """
    source = workspace.resolve_workspace_path(source_path)
    destination = workspace.resolve_workspace_path(destination_path)

    if not source.exists():
        raise FileNotFoundError(f"Source path '{source_path}' not found")

    if destination.exists():
        raise FileExistsError(f"Destination path '{destination_path}' already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        shutil.copy2(source, destination)
    elif source.is_dir():
        shutil.copytree(source, destination)
    else:
        raise ValueError(f"'{source_path}' is not a file or directory")

    return f"Copied '{source_path}' to '{destination_path}' successfully."


def snapshot_path(workspace: WorkspaceConfig, path: str) -> dict:
    """
        Takes a rollback snapshot of either a file or directory.
    """
    target = workspace.resolve_workspace_path(path)

    if not target.exists():
        return {
            "path": path,
            "existed": False,
            "kind": None,
            "content": None,
            "files": [],
            "dirs": []
        }

    if target.is_file():
        return {
            "path": path,
            "existed": True,
            "kind": "file",
            "content": target.read_bytes(),
            "files": [],
            "dirs": []
        }

    if target.is_dir():
        files = []
        dirs = []

        for item in sorted(target.rglob("*")):
            relative = str(item.relative_to(target))

            if item.is_dir():
                dirs.append(relative)

            elif item.is_file():
                files.append({
                    "relative_path": relative,
                    "content": item.read_bytes()
                })

        return {
            "path": path,
            "existed": True,
            "kind": "directory",
            "content": None,
            "files": files,
            "dirs": dirs
        }

    raise ValueError(f"Cannot snapshot '{path}'")


def rollback_path_snapshot(workspace: WorkspaceConfig, snapshot: dict):
    """
        Restores a file or directory from a rollback snapshot.
    """
    target = workspace.resolve_workspace_path(snapshot["path"])

    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    if not snapshot["existed"]:
        return

    if snapshot["kind"] == "file":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(snapshot["content"])
        return

    if snapshot["kind"] == "directory":
        target.mkdir(parents=True, exist_ok=True)

        for relative_dir in snapshot["dirs"]:
            (target / relative_dir).mkdir(parents=True, exist_ok=True)

        for file_info in snapshot["files"]:
            file_path = target / file_info["relative_path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(file_info["content"])

        return

    raise ValueError(f"Unknown snapshot kind: {snapshot['kind']}")


def snapshot_many_paths(workspace: WorkspaceConfig, paths: list[str]) -> dict:
    """
        Takes snapshots of multiple paths before a move/copy operation.
    """
    return {
        "snapshots": [
            snapshot_path(workspace, path)
            for path in paths
        ]
    }


def rollback_many_path_snapshots(workspace: WorkspaceConfig, snapshot_group: dict):
    """
        Restores multiple path snapshots.
    """
    for snapshot in reversed(snapshot_group["snapshots"]):
        rollback_path_snapshot(workspace, snapshot)