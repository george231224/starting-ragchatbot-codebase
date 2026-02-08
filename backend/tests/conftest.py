"""Shared fixtures for RAG system tests."""

import sys
import os
from unittest.mock import patch

import pytest

# Add backend directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from chromadb.config import Settings
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore


def _make_vector_store(max_results: int) -> VectorStore:
    """Create a VectorStore backed by an EphemeralClient (no disk)."""
    original_persistent = chromadb.PersistentClient

    def ephemeral_factory(path=None, settings=None):
        return chromadb.EphemeralClient(
            settings=settings or Settings(anonymized_telemetry=False)
        )

    with patch.object(chromadb, "PersistentClient", new=ephemeral_factory):
        store = VectorStore(
            chroma_path="unused",
            embedding_model="all-MiniLM-L6-v2",
            max_results=max_results,
        )
    return store


@pytest.fixture()
def sample_course_data():
    """Return a Course with 2 lessons and 3 CourseChunk objects."""
    course = Course(
        title="Introduction to Python",
        course_link="https://example.com/python",
        instructor="Jane Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Variables and Types",
                lesson_link="https://example.com/python/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Control Flow",
                lesson_link="https://example.com/python/lesson2",
            ),
        ],
    )

    chunks = [
        CourseChunk(
            content="Python is a high-level programming language known for its readability and versatility.",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables in Python are created when you assign a value. Python uses dynamic typing.",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Control flow in Python uses if, elif, and else statements for conditional execution.",
            course_title="Introduction to Python",
            lesson_number=2,
            chunk_index=2,
        ),
    ]

    return course, chunks


@pytest.fixture()
def ephemeral_vector_store():
    """VectorStore with max_results=5 (correct config), empty."""
    store = _make_vector_store(max_results=5)
    store.clear_all_data()
    return store


@pytest.fixture()
def buggy_vector_store():
    """VectorStore with max_results=0 (the bug), empty."""
    store = _make_vector_store(max_results=0)
    store.clear_all_data()
    return store


@pytest.fixture()
def seeded_store(sample_course_data):
    """Fresh VectorStore (max_results=5) pre-populated with sample data."""
    store = _make_vector_store(max_results=5)
    course, chunks = sample_course_data
    store.add_course_metadata(course)
    store.add_course_content(chunks)
    return store


@pytest.fixture()
def seeded_buggy_store(sample_course_data):
    """Fresh VectorStore (max_results=0) pre-populated with sample data."""
    store = _make_vector_store(max_results=0)
    course, chunks = sample_course_data
    store.add_course_metadata(course)
    store.add_course_content(chunks)
    return store
