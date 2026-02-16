"""API endpoint tests for the RAG system.

Defines a lightweight test app inline to avoid importing backend/app.py,
which mounts static files from a directory that doesn't exist in the test
environment.  The endpoint logic mirrors app.py exactly.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ---------------------------------------------------------------------------
# Inline test app – same endpoints as app.py, no static-file mount
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def _create_test_app(rag_system) -> FastAPI:
    """Build a FastAPI app with the same routes as app.py."""
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(mock_rag_system):
    """TestClient wired to a FastAPI app backed by the shared mock_rag_system."""
    app = _create_test_app(mock_rag_system)
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for the /api/query endpoint."""

    def test_query_with_session_id(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What is Python?", "session_id": "sess_42"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess_42"
        assert "Python" in body["answer"]
        assert len(body["sources"]) == 1
        assert body["sources"][0]["text"] == "Introduction to Python - Lesson 1"
        mock_rag_system.query.assert_called_once_with("What is Python?", "sess_42")

    def test_query_without_session_id_creates_session(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "Hello"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("Hello", "session_1")

    def test_query_returns_sources_with_links(self, client):
        resp = client.post("/api/query", json={"query": "Tell me about Python"})

        body = resp.json()
        assert body["sources"][0]["link"] == "https://example.com/python/lesson1"

    def test_query_missing_field_returns_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_query_server_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB connection lost")

        resp = client.post("/api/query", json={"query": "anything", "session_id": "s1"})

        assert resp.status_code == 500
        assert "DB connection lost" in resp.json()["detail"]

    def test_query_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("No results found.", [])

        resp = client.post("/api/query", json={"query": "obscure topic", "session_id": "s1"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"] == []
        assert body["answer"] == "No results found."


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint."""

    def test_get_courses(self, client):
        resp = client.get("/api/courses")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 2
        assert "Introduction to Python" in body["course_titles"]
        assert "Advanced Python" in body["course_titles"]

    def test_get_courses_server_error(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("ChromaDB error")

        resp = client.get("/api/courses")

        assert resp.status_code == 500
        assert "ChromaDB error" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestSessionEndpoint:
    """Tests for the /api/sessions/{session_id} endpoint."""

    def test_delete_session(self, client, mock_rag_system):
        resp = client.delete("/api/sessions/sess_42")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("sess_42")

    def test_delete_session_error(self, client, mock_rag_system):
        mock_rag_system.session_manager.clear_session.side_effect = KeyError("not found")

        resp = client.delete("/api/sessions/bad_id")

        assert resp.status_code == 500
