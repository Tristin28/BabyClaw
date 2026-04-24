import json
from pathlib import Path


CONFIG_PATH = Path("config/workspace_config.json")

def load_workspace_path() -> str:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            return data.get("workspace_root", "./workspace")

    return "./workspace"


def save_workspace_path(path: str):
    CONFIG_PATH.parent.mkdir(exist_ok=True)

    with open(CONFIG_PATH, "w") as f:
        json.dump({"workspace_root": path}, f, indent=2)