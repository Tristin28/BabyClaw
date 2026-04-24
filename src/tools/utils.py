from pathlib import Path

class WorkspaceConfig:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def set_root(self, new_root: str):
        new_path = Path(new_root).resolve()

        if not new_path.exists():
            raise FileNotFoundError(f"Workspace path '{new_root}' does not exist")

        if not new_path.is_dir():
            raise ValueError(f"Workspace path '{new_root}' is not a directory")

        self.root = new_path

def resolve_workspace_path(relative_path: str, workspace: WorkspaceConfig) -> Path:
    """
        Method is used to resolve a safe absolute path inside the workspace directory
        i.e. it grabs the path the system gives (can be combined either with subdirectories inside workspace or just the respective files)
        then it combines them with the respective absolute path of the workspace folder
    """

    resolved_path = (workspace.root / relative_path).resolve()

    if not resolved_path.is_relative_to(workspace.root):
        raise PermissionError("Access outside workspace is not allowed")

    return resolved_path