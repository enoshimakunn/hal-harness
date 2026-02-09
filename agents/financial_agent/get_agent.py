from agent import Agent
from tools import EDGARSearch, GoogleWebSearch, ParseHtmlPage, RetrieveInformation
from llm_wrapper import LiteLLMWrapper

from dataclasses import dataclass
from typing import List


@dataclass
class Parameters:
    model_name: str
    max_turns: int
    tools: List[str]
    temperature: float = 0.0
    max_tokens: int = 32000


def get_agent(parameters: Parameters) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    available_tools = {
        "google_web_search": GoogleWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
    }

    selected_tools = {}
    for tool in parameters.tools:
        if tool not in available_tools:
            raise Exception(
                f"Tool {tool} not found in tools. Available tools: {available_tools.keys()}"
            )
        selected_tools[tool] = available_tools[tool]()

    # Create LiteLLM wrapper
    model = LiteLLMWrapper(
        model=parameters.model_name,
        temperature=parameters.temperature,
        max_tokens=parameters.max_tokens,
    )

    agent = Agent(tools=selected_tools, llm=model, max_turns=parameters.max_turns)

    return agent
