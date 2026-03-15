import os
from typing import Dict

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.tools.task_tracker import TaskTrackerTool


def run(input: Dict[str, dict], **kwargs) -> Dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    # Initialize LLM
    llm = LLM(
        model=kwargs["model_name"],
        api_key=os.getenv("LLM_API_KEY"),
    )

    # Initialize Agent with tools
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
    )

    # Create Conversation
    workspace = os.getcwd()
    print(f"[openhands_agent] workspace={workspace}")
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
    )

    output_filename = f"solution_{task_id}.py"
    output_path = os.path.join(workspace, output_filename)
    print(f"[openhands_agent] expected_output_path={output_path}")
    if os.path.exists(output_path):
        print(f"[openhands_agent] removing_stale_output={output_path}")
        os.remove(output_path)
    try:
        print(f"[openhands_agent] pre_run_workspace_files={sorted(os.listdir(workspace))}")
    except Exception as exc:
        print(f"[openhands_agent] failed_to_list_workspace_before_run: {exc}")
    skill_context = kwargs.get("skill_content")
    skill_text = ""
    if isinstance(skill_context, str) and skill_context.strip():
        skill_text = f"\n\nSkill guidance:\n{skill_context.strip()}\n"

    prompt = f"""
    You are solving a USACO problem.

    Your task:
    - Write a complete and correct Python 3 solution.
    - Read from stdin, write to stdout.
    - Efficient for full constraints.

    CRITICAL INSTRUCTIONS:
    - Write the FINAL ANSWER to this exact path: "{output_path}"
    - The output file must be inside the current workspace, not elsewhere under /tmp or /private/tmp
    - Write the complete final solution in a single pass when possible
    - Create the file once with the full final solution if it does not exist
    - If it already exists, overwrite that exact file with the full final solution
    - Do not use repeated small string replacements on the solution file
    - Do not create alternate solution files or variants with different names
    - The file must contain ONLY valid Python code
    - No explanations, no markdown, no extra text
    - After the file is fully written, call finish immediately
    - Do not continue editing after the final file has been written
    - Do NOT print anything else outside the solution file

    Selected skill:
    {skill_text.strip() if skill_text else "No skill guidance provided."}

    Problem:
    {task["description"]}
    """.strip()
    print("Prompt for agent:")
    print(prompt)
    conversation.send_message(prompt)

    print(f"[openhands_agent] starting_conversation task_id={task_id}")
    try:
        conversation.run()
        print(f"[openhands_agent] conversation_completed task_id={task_id}")
    except Exception as exc:
        print(f"[openhands_agent] conversation_run_failed: {type(exc).__name__}: {exc}")
        raise

    try:
        print(f"[openhands_agent] post_run_workspace_files={sorted(os.listdir(workspace))}")
    except Exception as exc:
        print(f"[openhands_agent] failed_to_list_workspace_after_run: {exc}")

    candidate_output_files: list[str] = []
    for root, _, files in os.walk(workspace):
        for filename in files:
            if "solution" in filename.lower() or task_id in filename:
                candidate_output_files.append(
                    os.path.relpath(os.path.join(root, filename), workspace)
                )
    print(f"[openhands_agent] candidate_output_files={sorted(candidate_output_files)}")

    if not os.path.exists(output_path):
        print(
            f"[openhands_agent] missing_expected_output_file task_id={task_id} "
            f"expected={output_filename}"
        )
        raise RuntimeError(
            f"Agent did not produce expected output file: {output_filename}"
        )

    with open(output_path, "r", encoding="utf-8") as f:
        final_output = f.read().strip()

    if not final_output:
        raise RuntimeError("Output file is empty")

    return {task_id: final_output}
