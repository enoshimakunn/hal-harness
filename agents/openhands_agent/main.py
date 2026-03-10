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
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
    )

    output_filename = f"solution_{task_id}.py"
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
    - Write the FINAL ANSWER into a file named "{output_filename}"
    - The file must contain ONLY valid Python code
    - No explanations, no markdown, no extra text
    - Do NOT call finish
    - Do NOT print anything else
{skill_text}

    Problem:
    {task["description"]}
    """.strip()
    print("Prompt for agent:")
    print(prompt)
    conversation.send_message(prompt)

    conversation.run()

    output_path = os.path.join(workspace, output_filename)

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Agent did not produce expected output file: {output_filename}"
        )

    with open(output_path, "r", encoding="utf-8") as f:
        final_output = f.read().strip()

    if not final_output:
        raise RuntimeError("Output file is empty")

    return {task_id: final_output}
