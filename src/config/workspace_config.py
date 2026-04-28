import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "workspace_config.json"
DEFAULT_WORKSPACE = PROJECT_ROOT / "workspace"


def load_workspace_path() -> str:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            stored = data.get("workspace_root")
            if stored:
                return stored

    return str(DEFAULT_WORKSPACE)


def save_workspace_path(path) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "w") as f:
        json.dump({"workspace_root": str(path)}, f, indent=2)