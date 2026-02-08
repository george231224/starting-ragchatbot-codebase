"""Tests for AIGenerator with fully mocked Anthropic API.

No real API calls are made. We use mock Message, TextBlock, and ToolUseBlock
objects to simulate Claude responses.
"""

from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass
from typing import Any

import pytest
from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Lightweight mock response objects
# ---------------------------------------------------------------------------

@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class MockMessage:
    content: list
    stop_reason: str


def _make_text_response(text: str) -> MockMessage:
    return MockMessage(content=[MockTextBlock(text=text)], stop_reason="end_turn")


def _make_tool_response(tool_name: str, tool_input: dict, tool_id: str = "toolu_01") -> MockMessage:
    return MockMessage(
        content=[MockToolUseBlock(id=tool_id, name=tool_name, input=tool_input)],
        stop_reason="tool_use",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client():
    """Patch anthropic.Anthropic and return the mock client instance."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        client_instance = MagicMock()
        MockAnthropic.return_value = client_instance
        yield client_instance


@pytest.fixture()
def generator(mock_client):
    """AIGenerator wired to the mocked Anthropic client."""
    return AIGenerator(api_key="fake-key", model="claude-test")


# ---------------------------------------------------------------------------
# Direct text responses (no tools)
# ---------------------------------------------------------------------------

class TestDirectResponses:
    def test_direct_text_response(self, mock_client, generator):
        """Non-tool response returns text directly."""
        mock_client.messages.create.return_value = _make_text_response("Hello!")
        result = generator.generate_response(query="Hi")
        assert result == "Hello!"

    def test_no_tools_excludes_tools_from_params(self, mock_client, generator):
        """When no tools provided, 'tools' and 'tool_choice' should not appear."""
        mock_client.messages.create.return_value = _make_text_response("response")
        generator.generate_response(query="test")

        kwargs = mock_client.messages.create.call_args
        assert "tools" not in kwargs.kwargs
        assert "tool_choice" not in kwargs.kwargs


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_conversation_history_in_system_prompt(self, mock_client, generator):
        """History string is appended to system content."""
        mock_client.messages.create.return_value = _make_text_response("ok")
        generator.generate_response(query="q", conversation_history="User: hi\nAssistant: hello")

        kwargs = mock_client.messages.create.call_args
        system = kwargs.kwargs.get("system") or kwargs[1].get("system")
        assert "User: hi" in system
        assert "Previous conversation:" in system

    def test_no_history_uses_base_system_prompt(self, mock_client, generator):
        """No history → bare SYSTEM_PROMPT without 'Previous conversation'."""
        mock_client.messages.create.return_value = _make_text_response("ok")
        generator.generate_response(query="q")

        kwargs = mock_client.messages.create.call_args
        system = kwargs.kwargs.get("system") or kwargs[1].get("system")
        assert "Previous conversation:" not in system
        assert "AI assistant" in system


# ---------------------------------------------------------------------------
# Tool calling flow
# ---------------------------------------------------------------------------

class TestToolCalling:
    def test_tool_use_triggers_execution(self, mock_client, generator):
        """stop_reason=='tool_use' → tool_manager.execute_tool() called."""
        mock_client.messages.create.side_effect = [
            _make_tool_response("search_course_content", {"query": "Python"}),
            _make_text_response("Here are the results."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search results text"

        result = generator.generate_response(
            query="Tell me about Python",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python"
        )
        assert result == "Here are the results."

    def test_tools_included_in_api_params(self, mock_client, generator):
        """Tools appear in API kwargs with tool_choice: auto."""
        mock_client.messages.create.return_value = _make_text_response("ok")
        tools = [{"name": "search_course_content", "input_schema": {}}]
        generator.generate_response(query="q", tools=tools)

        kwargs = mock_client.messages.create.call_args
        assert kwargs.kwargs["tools"] == tools
        assert kwargs.kwargs["tool_choice"] == {"type": "auto"}

    def test_tool_result_message_format(self, mock_client, generator):
        """Follow-up call has correct 3-message structure."""
        tool_response = _make_tool_response("search_course_content", {"query": "x"}, tool_id="t1")
        mock_client.messages.create.side_effect = [
            tool_response,
            _make_text_response("final"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool output"

        generator.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        # Second call's messages should be: user, assistant (tool_use), user (tool_result)
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        # tool_result message content
        tool_result_content = messages[2]["content"]
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "t1"
        assert tool_result_content[0]["content"] == "tool output"

    def test_follow_up_call_excludes_tools(self, mock_client, generator):
        """Second API call has no 'tools' key."""
        mock_client.messages.create.side_effect = [
            _make_tool_response("search_course_content", {"query": "x"}),
            _make_text_response("done"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        generator.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" not in second_call.kwargs
