"""Tests for CourseSearchTool and ToolManager.

Uses real VectorStore with ephemeral ChromaDB — no mocking of the search layer.
"""

import pytest
from search_tools import CourseSearchTool, ToolManager


# ---------------------------------------------------------------------------
# CourseSearchTool – correct configuration (max_results=5)
# ---------------------------------------------------------------------------

class TestCourseSearchToolCorrectConfig:
    def test_execute_returns_results_with_valid_max_results(self, seeded_store):
        """Correct config → formatted results returned."""
        tool = CourseSearchTool(seeded_store)
        result = tool.execute(query="Python programming")
        assert "Introduction to Python" in result
        assert len(result) > 0

    def test_execute_with_course_name_filter(self, seeded_store):
        """Course name resolution + filtering works."""
        tool = CourseSearchTool(seeded_store)
        result = tool.execute(query="variables", course_name="Python")
        assert "Introduction to Python" in result

    def test_execute_with_lesson_number_filter(self, seeded_store):
        """Lesson number filtering works."""
        tool = CourseSearchTool(seeded_store)
        result = tool.execute(query="control flow", lesson_number=2)
        assert "Lesson 2" in result

    def test_execute_with_invalid_course_name(self, ephemeral_vector_store):
        """Returns 'No course found matching' when catalog is empty."""
        tool = CourseSearchTool(ephemeral_vector_store)
        result = tool.execute(query="anything", course_name="Nonexistent Course XYZ")
        assert "No course found matching" in result

    def test_execute_with_no_matching_content(self, ephemeral_vector_store):
        """Empty store → search error (no documents to query)."""
        tool = CourseSearchTool(ephemeral_vector_store)
        result = tool.execute(query="quantum physics")
        # An empty collection returns either no results or an error
        assert "No relevant content found" in result or "error" in result.lower()

    def test_last_sources_populated_after_search(self, seeded_store):
        """last_sources has entries with 'text' key after successful search."""
        tool = CourseSearchTool(seeded_store)
        tool.execute(query="Python programming")
        assert len(tool.last_sources) > 0
        assert "text" in tool.last_sources[0]

    def test_last_sources_contain_lesson_links(self, seeded_store):
        """Sources include valid lesson links."""
        tool = CourseSearchTool(seeded_store)
        tool.execute(query="variables types")
        has_link = any("link" in s for s in tool.last_sources)
        assert has_link, "Expected at least one source with a lesson link"


# ---------------------------------------------------------------------------
# CourseSearchTool – buggy configuration (max_results=0)
# ---------------------------------------------------------------------------

class TestCourseSearchToolBuggyConfig:
    def test_execute_fails_with_zero_max_results(self, seeded_buggy_store):
        """max_results=0 → search error mentioning zero/negative."""
        tool = CourseSearchTool(seeded_buggy_store)
        result = tool.execute(query="Python programming")
        # ChromaDB raises an error when n_results <= 0
        assert "Search error" in result or "error" in result.lower()

    def test_last_sources_empty_after_failed_search(self, seeded_buggy_store):
        """Bug → last_sources stays [] because search errors out."""
        tool = CourseSearchTool(seeded_buggy_store)
        tool.execute(query="Python programming")
        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:
    def test_execute_registered_tool(self, seeded_store):
        """ToolManager dispatches to the correct tool."""
        manager = ToolManager()
        tool = CourseSearchTool(seeded_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="Python")
        assert "Introduction to Python" in result

    def test_execute_unknown_tool(self):
        """Unknown tool → 'not found' message."""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_tool_definition_schema(self, seeded_store):
        """Tool definition has correct Anthropic tool schema shape."""
        tool = CourseSearchTool(seeded_store)
        defn = tool.get_tool_definition()

        assert defn["name"] == "search_course_content"
        assert "description" in defn
        assert "input_schema" in defn
        schema = defn["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "query" in schema["required"]
