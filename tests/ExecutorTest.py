from src.agents.execution.ExecutorAgent import ExecutorAgent
from src.llm.OllamaClient import OllamaClient
import copy 

#Method which will handle redundant test logic
def run_test(name, executor, plan_response):
    print(f"\n---- {name} ----")
    msg = executor.run( conversation_id=1,step_index=2, plan_response=plan_response)
    print(msg)

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
    llm = OllamaClient(model="qwen2.5:3b")  #Not actually used by Executor right now

    tool_registry = {
        "read_file": {
            "func": read_file,
            "input_map": {
                "file_id": "file_id"
            }
        },
        "summarise_txt": {
            "func": summarise_txt,
            "input_map": {
                "text": "source_step"
            }
        },
        "list_dir": {
            "func": list_dir,
            "input_map": {
                "path": "path"
            }
        }
    }

    executor = ExecutorAgent(llm_client=llm, tool_registry=tool_registry)

    base_plan = {
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

    run_test("VALID PLAN", executor, copy.deepcopy(base_plan))

    ''' Another Test but this type the plan will be invalid, so to check if Executor handles invalidity well '''
    #Testing bad dependency
    bad_dependency = copy.deepcopy(base_plan)
    bad_dependency["steps"][1]["depends_on"] = [99]
    run_test("BAD DEPENDENCY", executor, bad_dependency)

    #Testing missing file
    missing_file = copy.deepcopy(base_plan)
    missing_file["steps"][0]["args"]["file_id"] = "missing.txt"
    run_test("MISSING FILE", executor, missing_file)

    #Test bad source_step
    bad_source_step = copy.deepcopy(base_plan)
    bad_source_step["steps"][1]["args"]["source_step"] = 99
    run_test("BAD SOURCE STEP", executor, bad_source_step)


if __name__ == "__main__":
    main()