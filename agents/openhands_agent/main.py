"""
OpenHands agent for USACO problems
"""
import os
import tempfile
import shutil

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # Get model configuration
    model_name = kwargs["model_name"]
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize LLM with the specified model
    llm = LLM(
        model=model_name,
        api_key=api_key,
        base_url=None,
    )
    
    # Initialize agent with file and terminal tools
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=FileEditorTool.name),
            Tool(name=TerminalTool.name),
        ],
        system_prompt="""You are an expert competitive programmer solving USACO problems.
When given a problem, you should:
1. Read and understand the problem carefully
2. Write a complete Python solution in a file called 'solution.py'
3. The solution should read from stdin and write to stdout
4. Test your solution if possible
5. Return the final code from solution.py

Always save your solution to 'solution.py' file."""
    )
    
    results = {}
    
    for problem_id, problem_data in input.items():
        print(f"Processing problem: {problem_id}")
        
        # Create temporary workspace for this problem
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Initialize conversation with temporary workspace
                conversation = Conversation(agent=agent, workspace=tmpdir)
                
                # Extract problem description (USACO uses "description" key)
                problem_statement = problem_data.get("description", "")
                
                # Send task to agent
                task = f"""Solve this USACO competitive programming problem:

{problem_statement}

Write a complete Python solution in 'solution.py' that:
- Reads input from stdin
- Solves the problem efficiently  
- Writes output to stdout
- Handles all edge cases

After writing the solution, show me the contents of solution.py."""
                
                conversation.send_message(task)
                conversation.run()
                
                # Read the generated solution from the workspace
                solution_path = os.path.join(tmpdir, "solution.py")
                if os.path.exists(solution_path):
                    with open(solution_path, 'r') as f:
                        code = f.read()
                    results[problem_id] = code
                    print(f"Generated solution for {problem_id}")
                else:
                    print(f"No solution.py found for {problem_id}, using empty solution")
                    results[problem_id] = "# No solution generated"
                    
            except Exception as e:
                print(f"Error processing {problem_id}: {e}")
                results[problem_id] = f"# Error: {str(e)}"
    
    return results
