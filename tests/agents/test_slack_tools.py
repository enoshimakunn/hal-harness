"""Tests for Slack integration tools in the core agent."""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock smolagents before importing slack_tools, since smolagents may not be
# installed in the test environment. The @tool decorator is replaced with a
# passthrough so functions keep their original signature.
_mock_smolagents = MagicMock()
_mock_smolagents.tool = lambda fn: fn  # @tool becomes a no-op decorator
sys.modules.setdefault("smolagents", _mock_smolagents)

# Mock mcpadapt modules so that local imports inside get_slack_tools succeed.
# Individual tests override MCPAdapt/SmolAgentsAdapter behaviour as needed.
_mock_mcpadapt = MagicMock()
_mock_mcpadapt_core = MagicMock()
_mock_mcpadapt_adapter = MagicMock()
sys.modules.setdefault("mcpadapt", _mock_mcpadapt)
sys.modules.setdefault("mcpadapt.core", _mock_mcpadapt_core)
sys.modules.setdefault("mcpadapt.smolagents_adapter", _mock_mcpadapt_adapter)

# Add core_agent directory to path so we can import slack_tools directly
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "agents", "core_agent")
)

import slack_tools  # noqa: E402
from slack_tools import get_slack_tools, slack_realtime_search  # noqa: E402

MOCK_SLACK_SEARCH_RESPONSE = {
    "ok": True,
    "messages": {
        "matches": [
            {
                "channel": {"name": "general"},
                "username": "testuser",
                "text": "Hello world",
                "ts": "1234567890.123456",
            },
            {
                "channel": {"name": "random"},
                "username": "otheruser",
                "text": "Test message",
                "ts": "1234567891.654321",
            },
        ],
        "total": 2,
    },
}


class TestGetSlackToolsDisabled:
    """Tests for opt-in behavior when Slack is NOT configured."""

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_empty_list_when_no_token(self):
        assert get_slack_tools() == []

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": ""})
    def test_returns_empty_list_when_token_empty(self):
        assert get_slack_tools() == []

    @patch.dict(os.environ, {}, clear=True)
    def test_no_mcp_connection_attempted(self):
        """MCPAdapt should never be called when token is unset."""
        _mock_mcpadapt_core.MCPAdapt.reset_mock()
        get_slack_tools()
        _mock_mcpadapt_core.MCPAdapt.assert_not_called()


class TestGetSlackToolsWithMCP:
    """Tests for the MCP server connection path (mocked)."""

    def _setup_mock_client(self, mcp_tools=None):
        """Create and configure a mock MCPAdapt client."""
        if mcp_tools is None:
            mcp_tools = []
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mcp_tools)
        mock_client.__exit__ = Mock(return_value=False)
        return mock_client

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.atexit")
    def test_returns_mcp_tools_plus_search(self, mock_atexit):
        fake_tool_1 = Mock(name="slack_send_message")
        fake_tool_2 = Mock(name="slack_list_channels")
        mock_client = self._setup_mock_client([fake_tool_1, fake_tool_2])

        with patch.object(_mock_mcpadapt_core, "MCPAdapt", return_value=mock_client):
            tools = get_slack_tools()

        # Should contain 2 MCP tools + slack_realtime_search
        assert len(tools) == 3
        assert tools[0] is fake_tool_1
        assert tools[1] is fake_tool_2
        assert tools[2] is slack_realtime_search

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.atexit")
    def test_mcp_server_params_correct(self, mock_atexit):
        mock_client = self._setup_mock_client()

        with patch.object(
            _mock_mcpadapt_core, "MCPAdapt", return_value=mock_client
        ) as mock_cls:
            get_slack_tools()

        call_args = mock_cls.call_args
        server_params = call_args[0][0]
        assert server_params["url"] == "https://mcp.slack.com/mcp"
        assert server_params["transport"] == "streamable-http"
        assert server_params["headers"]["Authorization"] == "Bearer xoxp-test-token"

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    def test_atexit_handler_registered(self):
        mock_client = self._setup_mock_client()

        with patch.object(_mock_mcpadapt_core, "MCPAdapt", return_value=mock_client):
            with patch("slack_tools.atexit.register") as mock_register:
                get_slack_tools()

        mock_register.assert_called_once_with(
            mock_client.__exit__, None, None, None
        )

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.atexit")
    def test_mcp_connection_failure_falls_back(self, mock_atexit):
        """On MCP connection failure, should still return the search tool."""
        with patch.object(
            _mock_mcpadapt_core,
            "MCPAdapt",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.warns(UserWarning, match="Failed to connect"):
                tools = get_slack_tools()

        assert len(tools) == 1
        assert tools[0] is slack_realtime_search


class TestGetSlackToolsMcpadaptMissing:
    """Tests for graceful degradation when mcpadapt is not installed."""

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    def test_returns_search_tool_only_when_mcpadapt_missing(self):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name in ("mcpadapt.core", "mcpadapt.smolagents_adapter"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.warns(UserWarning, match="mcpadapt is not installed"):
                tools = get_slack_tools()

        assert len(tools) == 1
        assert tools[0] is slack_realtime_search

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    def test_logs_warning_when_mcpadapt_missing(self):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name in ("mcpadapt.core", "mcpadapt.smolagents_adapter"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.warns(UserWarning, match="mcpadapt is not installed"):
                get_slack_tools()


class TestSlackRealtimeSearch:
    """Tests for the slack_realtime_search @tool function."""

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.requests.get")
    def test_search_returns_formatted_messages(self, mock_get):
        mock_get.return_value = Mock(json=Mock(return_value=MOCK_SLACK_SEARCH_RESPONSE))

        result = slack_realtime_search("hello")

        assert "[#general] testuser" in result
        assert "Hello world" in result
        assert "[#random] otheruser" in result
        assert "Test message" in result
        assert "Found 2 results" in result

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.requests.get")
    def test_search_with_custom_count(self, mock_get):
        mock_get.return_value = Mock(json=Mock(return_value=MOCK_SLACK_SEARCH_RESPONSE))

        slack_realtime_search("hello", count=5)

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["count"] == 5

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.requests.get")
    def test_search_auth_header_correct(self, mock_get):
        mock_get.return_value = Mock(json=Mock(return_value=MOCK_SLACK_SEARCH_RESPONSE))

        slack_realtime_search("hello")

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer xoxp-test-token"

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.requests.get")
    def test_search_api_error_returns_error_message(self, mock_get):
        mock_get.return_value = Mock(
            json=Mock(return_value={"ok": False, "error": "invalid_auth"})
        )

        result = slack_realtime_search("hello")

        assert "Error" in result
        assert "invalid_auth" in result

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.requests.get")
    def test_search_network_error_returns_error_message(self, mock_get):
        import requests as req

        mock_get.side_effect = req.ConnectionError("Connection refused")

        result = slack_realtime_search("hello")

        assert "Error" in result
        assert "connect" in result.lower()

    @patch.dict(os.environ, {}, clear=True)
    def test_search_without_token_returns_error(self):
        result = slack_realtime_search("hello")
        assert "Error" in result
        assert "SLACK_USER_TOKEN" in result


class TestSlackToolsEndToEnd:
    """Simulates the real integration scenario in main.py."""

    @patch.dict(os.environ, {}, clear=True)
    def test_core_tools_unchanged_without_slack(self):
        original_tools = [Mock() for _ in range(9)]
        core_tools = list(original_tools)

        core_tools.extend(get_slack_tools())

        assert len(core_tools) == 9

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.atexit")
    def test_core_tools_extended_with_slack(self, mock_atexit):
        fake_mcp_tools = [Mock() for _ in range(3)]
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=fake_mcp_tools)
        mock_client.__exit__ = Mock(return_value=False)

        original_tools = [Mock() for _ in range(9)]
        core_tools = list(original_tools)

        with patch.object(_mock_mcpadapt_core, "MCPAdapt", return_value=mock_client):
            core_tools.extend(get_slack_tools())

        # 9 original + 3 MCP + 1 search = 13
        assert len(core_tools) == 13

    @patch.dict(os.environ, {"SLACK_USER_TOKEN": "xoxp-test-token"})
    @patch("slack_tools.atexit")
    def test_slack_tools_are_smolagents_compatible(self, mock_atexit):
        """MCP tools must have smolagents interface; search tool must be callable."""
        fake_mcp_tool = Mock()
        fake_mcp_tool.name = "slack_send"
        fake_mcp_tool.description = "Send a Slack message"
        fake_mcp_tool.inputs = {"channel": {"type": "string"}}
        fake_mcp_tool.output_type = "string"
        fake_mcp_tool.forward = Mock()

        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=[fake_mcp_tool])
        mock_client.__exit__ = Mock(return_value=False)

        with patch.object(_mock_mcpadapt_core, "MCPAdapt", return_value=mock_client):
            tools = get_slack_tools()

        # MCP tools (from adapter) have the full smolagents Tool interface
        mcp_tool = tools[0]
        assert hasattr(mcp_tool, "name")
        assert hasattr(mcp_tool, "description")
        assert hasattr(mcp_tool, "inputs")
        assert hasattr(mcp_tool, "output_type")
        assert callable(getattr(mcp_tool, "forward", None))

        # The search tool is callable (in production, @tool adds the full
        # interface; here the decorator is mocked as a no-op passthrough)
        search = tools[-1]
        assert callable(search)
        assert search is slack_realtime_search
