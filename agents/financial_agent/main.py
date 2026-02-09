# HAL-compatible Financial Agent wrapper
#
# This agent uses litellm for LLM calls - no external model_library dependency.
# All agent logic is in this directory - fully self-contained.

import asyncio
import os
import sys

# Add this directory to path for local imports
agent_dir = os.path.dirname(os.path.abspath(__file__))
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

from dotenv import load_dotenv

load_dotenv()

# Import from local modules
from get_agent import get_agent, Parameters


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    HAL-compatible agent function for financial research questions.

    Args:
        input: Dictionary mapping question IDs to question data.
               Each question data dict has keys: question, answer, question_type, etc.
        **kwargs: Additional arguments:
            - model_name: LLM model to use (default: openai/gpt-4o)
            - max_turns: Maximum agent turns (default: 20)
            - tools: List of tools to use (default: all)

    Returns:
        Dictionary mapping question IDs to agent answer strings.
    """
    # Extract kwargs with defaults
    model_name = kwargs.get("model_name", "openai/gpt-4o")
    max_turns = int(kwargs.get("max_turns", 20))
    temperature = float(kwargs.get("temperature", 0.0))
    max_tokens = int(kwargs.get("max_tokens", 16000))
    tools = kwargs.get("tools", [
        "google_web_search",
        "retrieve_information",
        "parse_html_page",
        "edgar_search",
    ])

    # Handle tools as string (comma-separated) or list
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",")]

    # Create agent parameters
    parameters = Parameters(
        model_name=model_name,
        max_turns=max_turns,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Initialize agent
    agent = get_agent(parameters)

    # Process each question
    results = {}

    for question_id, question_data in input.items():
        question = question_data.get("question", "")
        if not question:
            results[question_id] = ""
            continue

        try:
            # Run the agent asynchronously
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running (e.g., in Jupyter), use nest_asyncio or run_until_complete
                    import nest_asyncio
                    nest_asyncio.apply()
                answer, metadata = loop.run_until_complete(agent.run(question))
            except RuntimeError:
                # No event loop running, create a new one
                answer, metadata = asyncio.run(agent.run(question))
            
            results[question_id] = answer if answer else ""
        except Exception as e:
            # Record error as answer
            results[question_id] = f"Error: {str(e)}"

    return results
