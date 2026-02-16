"""End-to-end tests for RAGSystem.query().

Uses real VectorStore (ephemeral ChromaDB) + mocked Anthropic API.
Tests the full orchestration including tool dispatch and source tracking.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from session_manager import SessionManager
from rag_system import RAGSystem

# ---------------------------------------------------------------------------
# Mock response helpers (same lightweight approach as test_ai_generator)
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


def _text(text: str) -> MockMessage:
    return MockMessage(content=[MockTextBlock(text=text)], stop_reason="end_turn")


def _tool_call(name: str, inp: dict, tid: str = "toolu_01") -> MockMessage:
    return MockMessage(
        content=[MockToolUseBlock(id=tid, name=name, input=inp)],
        stop_reason="tool_use",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rag_system(seeded_store):
    """RAGSystem wired to a seeded ephemeral VectorStore and mocked AI client."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        # Build a minimal Config-like object
        class FakeConfig:
            ANTHROPIC_API_KEY = "fake-key"
            ANTHROPIC_MODEL = "claude-test"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 5
            MAX_HISTORY = 2
            CHROMA_PATH = "unused"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"

        # We can't call RAGSystem.__init__ directly because it creates its own
        # VectorStore with PersistentClient. Instead, assemble manually.
        system = object.__new__(RAGSystem)
        system.config = FakeConfig()
        system.vector_store = seeded_store
        system.ai_generator = AIGenerator(api_key="fake-key", model="claude-test")
        system.session_manager = SessionManager(max_history=2)
        system.tool_manager = ToolManager()
        system.search_tool = CourseSearchTool(seeded_store)
        system.tool_manager.register_tool(system.search_tool)
        system.outline_tool = CourseOutlineTool(seeded_store)
        system.tool_manager.register_tool(system.outline_tool)

        yield system, mock_client


@pytest.fixture()
def buggy_rag_system(seeded_buggy_store):
    """RAGSystem with max_results=0 VectorStore (the bug)."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        system = object.__new__(RAGSystem)

        class FakeConfig:
            ANTHROPIC_API_KEY = "fake-key"
            ANTHROPIC_MODEL = "claude-test"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 0
            MAX_HISTORY = 2
            CHROMA_PATH = "unused"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"

        system.config = FakeConfig()
        system.vector_store = seeded_buggy_store
        system.ai_generator = AIGenerator(api_key="fake-key", model="claude-test")
        system.session_manager = SessionManager(max_history=2)
        system.tool_manager = ToolManager()
        system.search_tool = CourseSearchTool(seeded_buggy_store)
        system.tool_manager.register_tool(system.search_tool)
        system.outline_tool = CourseOutlineTool(seeded_buggy_store)
        system.tool_manager.register_tool(system.outline_tool)

        yield system, mock_client


# ---------------------------------------------------------------------------
# Tests – correct config
# ---------------------------------------------------------------------------


class TestRAGQueryCorrectConfig:
    def test_tool_call_produces_response_with_sources(self, rag_system):
        """Full flow: question → tool → search → answer + sources."""
        system, mock_client = rag_system

        mock_client.messages.create.side_effect = [
            _tool_call("search_course_content", {"query": "Python programming"}),
            _text("Python is a versatile language."),
        ]

        response, sources = system.query("What is Python?")
        assert response == "Python is a versatile language."
        assert len(sources) > 0

    def test_sources_contain_lesson_info(self, rag_system):
        """Sources have text entries with lesson info."""
        system, mock_client = rag_system

        mock_client.messages.create.side_effect = [
            _tool_call("search_course_content", {"query": "variables"}),
            _text("Variables store data."),
        ]

        _, sources = system.query("Tell me about variables")
        assert len(sources) > 0
        assert any("Lesson" in s["text"] for s in sources)

    def test_direct_response_returns_empty_sources(self, rag_system):
        """No tool use → empty sources."""
        system, mock_client = rag_system
        mock_client.messages.create.return_value = _text("Hello there!")

        response, sources = system.query("Hi")
        assert response == "Hello there!"
        assert sources == []

    def test_sources_reset_between_queries(self, rag_system):
        """Sources don't leak across queries."""
        system, mock_client = rag_system

        # First query uses tools
        mock_client.messages.create.side_effect = [
            _tool_call("search_course_content", {"query": "Python"}),
            _text("Answer 1"),
        ]
        _, sources1 = system.query("Q1")
        assert len(sources1) > 0

        # Second query is direct (no tool)
        mock_client.messages.create.side_effect = None
        mock_client.messages.create.return_value = _text("Answer 2")
        _, sources2 = system.query("Q2")
        assert sources2 == []

    def test_session_history_updated_after_query(self, rag_system):
        """Session contains the exchange after query."""
        system, mock_client = rag_system
        mock_client.messages.create.return_value = _text("response")

        sid = system.session_manager.create_session()
        system.query("my question", session_id=sid)

        history = system.session_manager.get_conversation_history(sid)
        assert "my question" in history
        assert "response" in history

    def test_query_without_session_still_returns_response(self, rag_system):
        """session_id=None works fine."""
        system, mock_client = rag_system
        mock_client.messages.create.return_value = _text("no session")

        response, sources = system.query("Hello", session_id=None)
        assert response == "no session"


# ---------------------------------------------------------------------------
# Tests – buggy config (max_results=0)
# ---------------------------------------------------------------------------


class TestRAGQueryBuggyConfig:
    def test_search_error_in_tool_result(self, buggy_rag_system):
        """MAX_RESULTS=0 → tool_result contains ChromaDB error string."""
        system, mock_client = buggy_rag_system

        # Capture what the tool returns to the second API call
        final_response = _text("Sorry, search failed.")
        mock_client.messages.create.side_effect = [
            _tool_call("search_course_content", {"query": "Python"}),
            final_response,
        ]

        system.query("What is Python?")

        # The second API call should contain the error from the tool
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_result_msg = messages[2]  # user message with tool_result
        tool_content = tool_result_msg["content"][0]["content"]
        assert "error" in tool_content.lower() or "Search error" in tool_content

    def test_no_sources_with_zero_max_results(self, buggy_rag_system):
        """MAX_RESULTS=0 → sources always [] because search errors out."""
        system, mock_client = buggy_rag_system

        mock_client.messages.create.side_effect = [
            _tool_call("search_course_content", {"query": "Python"}),
            _text("Error occurred."),
        ]

        _, sources = system.query("What is Python?")
        assert sources == []
