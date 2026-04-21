'''
        All followins tools can only read files (or list directories) based on the respective folder which is workspace, 
        i.e. it will only be able to access files if they are on the same directory. Hence this will enforce workspace sandboxing so the respective system it runs on
        is untouched.
    '''
from src.tools.utils import resolve_workspace_path

def read_file(path: str) -> str:
    """
        Reads a text file safely from inside the workspace sandbox.
    """
    file_path = resolve_workspace_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    if not file_path.is_file():
        raise ValueError(f"'{path}' is not a file")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def list_dir(path: str = ".") -> list[str]:
    """
        Lists the contents of a directory safely from inside the workspace sandbox.
        Further point i set the local variable path to "." as a default value so the system can 
        know about anything which is in the workspace folder too and not just subdirectories
    """
    dir_path = resolve_workspace_path(path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{path}' not found")

    if not dir_path.is_dir():
        raise ValueError(f"'{path}' is not a directory")

    return [item.name for item in dir_path.iterdir()]