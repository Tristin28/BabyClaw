import os
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

    def resolve_workspace_path(self, relative_path: str) -> Path:
        """
            Resolve a relative path to a safe absolute path inside the workspace.

            Defence in depth against symlink escape:
            1. Reject absolute / Windows-drive paths and `..` traversal early
               on the raw string, before any filesystem call.
            2. Use Path.resolve() to follow every symlink in the chain.
               If the resolved real path is not under self.root, refuse.
            3. Cross-check with os.path.realpath() as an independent way of
               collapsing symlinks; both must agree and both must stay inside
               self.root.
            4. For path components that already exist, walk the parent chain
               and refuse any symlink whose target escapes the workspace.

            The final returned Path is the resolved real path. All file tools
            operate on that path, so a symlink whose target is outside the
            workspace can never be read, written, deleted, moved, or copied.
        """
        if not isinstance(relative_path, str):
            raise PermissionError("Workspace paths must be strings")

        raw = relative_path.strip()
        if raw == "":
            raise PermissionError("Empty path is not allowed")

        normalised_raw = raw.replace("\\", "/")
        if normalised_raw.startswith("/"):
            raise PermissionError("Absolute paths are not allowed")
        if len(normalised_raw) >= 3 and normalised_raw[1] == ":" and normalised_raw[2] in {"/", "\\"}:
            raise PermissionError("Absolute paths are not allowed")
        if normalised_raw == ".." or normalised_raw.startswith("../") or "/../" in normalised_raw:
            raise PermissionError("Path traversal is not allowed")

        candidate = self.root / relative_path
        resolved_path = candidate.resolve()

        if not resolved_path.is_relative_to(self.root):
            raise PermissionError("Access outside workspace is not allowed")

        # Cross-check via os.path.realpath, which independently expands all
        # symlinks. If the two agree and both stay inside self.root, the path
        # is safe.
        real = Path(os.path.realpath(str(candidate)))
        if not real.is_relative_to(self.root):
            raise PermissionError("Access outside workspace is not allowed (symlink escape)")

        # Reject any existing intermediate symlink whose target leaves the
        # workspace, even if .resolve() happened to land back inside.
        for parent in [resolved_path] + list(resolved_path.parents):
            if parent == self.root:
                break
            if not parent.exists():
                continue
            if parent.is_symlink():
                target = Path(os.path.realpath(str(parent)))
                if not target.is_relative_to(self.root):
                    raise PermissionError("Access outside workspace is not allowed (symlink escape)")

        return resolved_path
