from src.Agents.Planning.PlanCompiler import PlanCompiler
from src.tools.tool_description import PLANNER_TOOL_DESCRIPTIONS


def run_test(name, plan, should_pass):
    compiler = PlanCompiler(PLANNER_TOOL_DESCRIPTIONS)

    print(f"\nTEST: {name}")

    try:
        compiled = compiler.compile(plan)

        if should_pass:
            print("PASSED as expected")
            print(compiled)
        else:
            print("FAILED: plan should have been rejected but passed")
            print(compiled)

    except Exception as e:
        if should_pass:
            print("FAILED: plan should have passed but was rejected")
            print(e)
        else:
            print("REJECTED as expected")
            print(e)


valid_plan = {
    "goal": "Read hello.txt and create BabyClaw with same content",
    "steps": [
        {
            "id": 1,
            "tool": "read_file",
            "args": {"path": "hello.txt"}
        },
        {
            "id": 2,
            "tool": "create_file",
            "args": {"path": "BabyClaw", "content_step": 1}
        }
    ],
    "planning_rationale": "Read the source file, then use its content to create the new file."
}


fake_step_string_plan = {
    "goal": "Bad fake step string",
    "steps": [
        {
            "id": 1,
            "tool": "read_file",
            "args": {"path": "hello.txt"}
        },
        {
            "id": 2,
            "tool": "create_file",
            "args": {"path": "BabyClaw", "content": "text_step:1"}
        }
    ],
    "planning_rationale": "Bad plan."
}


wrong_direct_type_plan = {
    "goal": "Bad direct type",
    "steps": [
        {
            "id": 1,
            "tool": "read_file",
            "args": {"path": "hello.txt"}
        },
        {
            "id": 2,
            "tool": "summarise_txt",
            "args": {"text": 1}
        }
    ],
    "planning_rationale": "Bad plan."
}


future_dependency_plan = {
    "goal": "Bad future dependency",
    "steps": [
        {
            "id": 1,
            "tool": "summarise_txt",
            "args": {"text_step": 2}
        },
        {
            "id": 2,
            "tool": "read_file",
            "args": {"path": "hello.txt"}
        }
    ],
    "planning_rationale": "Bad plan."
}


unknown_arg_plan = {
    "goal": "Bad unknown arg",
    "steps": [
        {
            "id": 1,
            "tool": "read_file",
            "args": {"filename": "hello.txt"}
        }
    ],
    "planning_rationale": "Bad plan."
}


run_test("valid content_step dependency", valid_plan, should_pass=True)
run_test("reject fake step string", fake_step_string_plan, should_pass=False)
run_test("reject text integer", wrong_direct_type_plan, should_pass=False)
run_test("reject future dependency", future_dependency_plan, should_pass=False)
run_test("reject unknown arg", unknown_arg_plan, should_pass=False)