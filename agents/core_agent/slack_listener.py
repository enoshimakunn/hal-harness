"""Slack bot that listens for @mentions and responds using the core agent.

Usage:
    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    export SLACK_USER_TOKEN=xoxp-...   # optional, for Slack MCP tools
    python slack_listener.py [--model openai/gpt-4o]

Requires a Slack app with Socket Mode enabled and the following bot scopes:
    app_mentions:read, chat:write, channels:history, im:history
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import smolagents.models
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    PythonInterpreterTool,
    VisitWebpageTool,
)

from slack_tools import get_slack_tools  # noqa: E402


# Same monkey-patch as main.py for GPT-5 / reasoning model compat
def supports_stop_parameter(model_id: str) -> bool:
    model_name = model_id.split("/")[-1]
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*|gpt-5[-\d]*)$"
    return not re.match(pattern, model_name)


smolagents.models.supports_stop_parameter = supports_stop_parameter

# Reuse the same tools from main.py that make sense for a chat bot.
# Intentionally excludes: execute_bash, edit_file (too dangerous from Slack),
# TextInspectorTool (needs model at construction), CustomFinalAnswerTool (benchmark-specific).
CHAT_TOOLS = [
    DuckDuckGoSearchTool(),
    VisitWebpageTool(),
    PythonInterpreterTool(),
    *get_slack_tools(),
]

AUTHORIZED_IMPORTS = [
    "requests", "json", "os", "re", "math", "datetime", "csv",
    "pandas", "numpy", "bs4", "sympy", "fractions",
]


def build_agent(model_name: str) -> CodeAgent:
    model = LiteLLMModel(model_id=model_name)
    return CodeAgent(
        tools=CHAT_TOOLS,
        model=model,
        max_steps=20,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
    )


def main():
    parser = argparse.ArgumentParser(description="Slack bot powered by the core agent")
    parser.add_argument("--model", default="openai/gpt-4o", help="LiteLLM model ID")
    args = parser.parse_args()

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        print("Error: SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set.")
        sys.exit(1)

    app = App(token=bot_token)
    agent = build_agent(args.model)

    @app.event("app_mention")
    def handle_mention(event, say):
        user_msg = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
        thread_ts = event.get("thread_ts") or event.get("ts")

        if not user_msg:
            say("Hey! Ask me something.", thread_ts=thread_ts)
            return

        say("On it, thinking...", thread_ts=thread_ts)

        try:
            response = agent.run(user_msg)
            reply = str(response)
        except Exception as exc:
            reply = f"Sorry, I hit an error: {exc}"

        # Slack has a 4000-char limit per message
        if len(reply) > 3900:
            reply = reply[:3900] + "\n... (truncated)"

        say(reply, thread_ts=thread_ts)

    print(f"Starting Slack bot with model={args.model}...")
    SocketModeHandler(app, app_token).start()


if __name__ == "__main__":
    main()
