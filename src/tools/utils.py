from pathlib import Path
'''
    Fixing the path to the workspace folder which is alongside the agent's logic.
    Further improvement is to have this path be configured by the user rather than being a fixed global var (i.e. through coord) 
    in order for the agent system to access any folder (sandbox) the user wants it to access
'''
WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "workspace" 

def resolve_workspace_path(relative_path: str) -> Path:
    """
        Method is used to resolve a safe absolute path inside the workspace directory
        i.e. it grabs the path the system gives (can be combined either with subdirectories inside workspace or just the respective files)
        then it combines them with the respective absolute path of the workspace folder
    """

    resolved_path = (WORKSPACE_ROOT / relative_path).resolve()

    if not resolved_path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise PermissionError("Access outside workspace is not allowed")

    return resolved_path