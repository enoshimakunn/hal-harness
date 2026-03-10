"""Slack integration for the core agent via MCP server + real-time search API.

Enabled when SLACK_USER_TOKEN environment variable is set.
Uses Slack's hosted MCP server (https://mcp.slack.com/mcp) for most operations
and a direct API call for real-time message search.
"""

import atexit
import os
import warnings

import requests
from smolagents import tool


def _get_slack_token() -> str | None:
    """Return the Slack user token if set and non-empty, else None."""
    return os.environ.get("SLACK_USER_TOKEN") or None


@tool
def slack_realtime_search(query: str, count: int = 10) -> str:
    """Search Slack messages in real time using the Slack Web API.

    Args:
        query: The search query string for finding Slack messages.
        count: Maximum number of results to return (default 10).

    Returns:
        Formatted search results or an error message.
    """
    token = _get_slack_token()
    if not token:
        return "Error: SLACK_USER_TOKEN environment variable is not set."

    try:
        resp = requests.get(
            "https://slack.com/api/search.messages",
            headers={"Authorization": f"Bearer {token}"},
            params={"query": query, "count": count},
            timeout=30,
        )
        data = resp.json()
    except requests.ConnectionError as exc:
        return f"Error: Could not connect to Slack API: {exc}"
    except Exception as exc:
        return f"Error: Slack API request failed: {exc}"

    if not data.get("ok"):
        return f"Error: Slack API returned error: {data.get('error', 'unknown error')}"

    matches = data.get("messages", {}).get("matches", [])
    if not matches:
        return f"No messages found for query: {query}"

    lines = []
    for msg in matches:
        channel = msg.get("channel", {}).get("name", "unknown")
        user = msg.get("username", "unknown")
        text = msg.get("text", "")
        ts = msg.get("ts", "")
        lines.append(f"[#{channel}] {user} ({ts}): {text}")

    total = data.get("messages", {}).get("total", len(matches))
    header = f"Found {total} results (showing {len(matches)}):"
    return header + "\n" + "\n".join(lines)


def get_slack_tools() -> list:
    """Return Slack tools if SLACK_USER_TOKEN is configured, else an empty list.

    Attempts to connect to Slack's hosted MCP server via mcpadapt for full
    Slack tool coverage. Falls back to the search-only tool if mcpadapt is
    unavailable or the MCP connection fails.
    """
    token = _get_slack_token()
    if not token:
        return []

    tools: list = []

    try:
        from mcpadapt.core import MCPAdapt
        from mcpadapt.smolagents_adapter import SmolAgentsAdapter

        client = MCPAdapt(
            {
                "url": "https://mcp.slack.com/mcp",
                "transport": "streamable-http",
                "headers": {"Authorization": f"Bearer {token}"},
            },
            SmolAgentsAdapter(),
        )
        mcp_tools = client.__enter__()
        tools.extend(mcp_tools)
        atexit.register(client.__exit__, None, None, None)
    except ImportError:
        warnings.warn(
            "mcpadapt is not installed. Only the slack_realtime_search tool will "
            "be available. Install mcpadapt for full Slack MCP tool support.",
            stacklevel=2,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to connect to Slack MCP server: {exc}. Only the "
            "slack_realtime_search tool will be available.",
            stacklevel=2,
        )

    tools.append(slack_realtime_search)
    return tools
