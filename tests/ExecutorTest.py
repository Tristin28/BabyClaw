from src.Agents.ExecutorAgent import ExecutorAgent
from src.OllamaClient import OllamaClient

'''
    Fake tools for now, as they are used for testing and even giving it fake files and folders
'''

def read_file(file_id: str) -> str:
    fake_filesystem = {
        "document.txt": "Machine learning is a field of artificial intelligence."
    }

    if file_id not in fake_filesystem:
        raise FileNotFoundError(f"File '{file_id}' not found")

    return fake_filesystem[file_id]

def summarise_txt(text: str) -> str:
    return f"Summary: {text[:30]}..."

def list_dir(path: str) -> list[str]:
    fake_dirs = {
        "workspace": ["document.txt", "notes.txt"]
    }

    if path not in fake_dirs:
        raise FileNotFoundError(f"Directory '{path}' not found")

    return fake_dirs[path]


def main():
    llm = OllamaClient(model="qwen2.5:3b")  # not actually used by Executor right now

    tool_registry = {
        "read_file": {
            "func": read_file,
            "description": "Reads the contents of a text file",
            "input_map": {
                "file_id": "file_id"
            }
        },
        "summarise_txt": {
            "func": summarise_txt,
            "description": "Summarises text from a previous step",
            "input_map": {
                "text": "source_step"
            }
        },
        "list_dir": {
            "func": list_dir,
            "description": "Lists files in a directory",
            "input_map": {
                "path": "path"
            }
        }
    }

    executor = ExecutorAgent(llm_client=llm, tool_registry=tool_registry)

    plan_response = {
        "goal": "Summarise document and inspect workspace",
        "steps": [
            {
                "id": 1,
                "tool": "read_file",
                "args": {
                    "file_id": "document.txt"
                },
                "depends_on": []
            },
            {
                "id": 2,
                "tool": "summarise_txt",
                "args": {
                    "source_step": 1
                },
                "depends_on": [1]
            },
            {
                "id": 3,
                "tool": "list_dir",
                "args": {
                    "path": "workspace"
                },
                "depends_on": []
            }
        ],
        "planning_rationale": "Read first, then summarise; listing workspace can run independently."
    }

    print("---- RAW EXECUTION RESPONSE ----")
    raw_response = executor.execute_plan(plan_response)
    print(raw_response)

    print("\n---- MESSAGE RESPONSE ----")
    msg = executor.run(conversation_id=1, step_index=2, plan_response=plan_response)
    print(msg)


if __name__ == "__main__":
    main()