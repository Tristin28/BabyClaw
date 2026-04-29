from src.core.workflow.ExecutionVerifier import ExecutionVerifier
from src.tools.utils import WorkspaceConfig


def make_verifier(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    return ExecutionVerifier(
        workspace_config=WorkspaceConfig(root=str(workspace_root))
    ), workspace_root


def test_execution_verifier_accepts_created_file_with_expected_content(tmp_path):
    verifier, workspace_root = make_verifier(tmp_path)
    target = workspace_root / "hello.txt"
    target.write_text("hello", encoding="utf-8")

    result = verifier.verify({
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "resolved_args": {
                    "path": "hello.txt",
                    "content": "hello"
                }
            }
        ]
    })

    assert result["accepted"] is True
    assert result["issues"] == []


def test_execution_verifier_rejects_missing_created_file(tmp_path):
    verifier, _ = make_verifier(tmp_path)

    result = verifier.verify({
        "execution_trace": [
            {
                "id": 1,
                "tool": "create_file",
                "status": "completed",
                "resolved_args": {
                    "path": "missing.txt",
                    "content": "hello"
                }
            }
        ]
    })

    assert result["accepted"] is False
    assert any("missing.txt" in issue for issue in result["issues"])


def test_execution_verifier_rejects_written_file_with_wrong_content(tmp_path):
    verifier, workspace_root = make_verifier(tmp_path)
    target = workspace_root / "hello.txt"
    target.write_text("wrong", encoding="utf-8")

    result = verifier.verify({
        "execution_trace": [
            {
                "id": 1,
                "tool": "write_file",
                "status": "completed",
                "resolved_args": {
                    "path": "hello.txt",
                    "content": "hello"
                }
            }
        ]
    })

    assert result["accepted"] is False
    assert any("final content differs" in issue for issue in result["issues"])
