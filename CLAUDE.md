# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Security Rules

- **NEVER read `.env` files or any files containing secrets/credentials**
- Do not read, cat, or access `.env`, `.env.local`, `.env.production`, or similar files
- If you need to check environment variable names, refer to `.env.example` instead

## Build and Run Commands

```bash
# Start the application (from backend directory, use Python 3.12 for PyTorch compatibility)
uv run --python 3.12 uvicorn app:app --reload --port 8000

# Install dependencies
uv sync
```

The app runs at `http://localhost:8000` with API docs at `/docs`.

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system for course material Q&A.

```
Frontend (Vanilla JS)
       │
       ▼
FastAPI Backend (/api/query, /api/courses)
       │
       ▼
RAGSystem (orchestrator)
    ├── DocumentProcessor → parses course files, chunks text
    ├── VectorStore → ChromaDB with two collections:
    │     ├── course_catalog (course metadata, name resolution)
    │     └── course_content (text chunks for search)
    ├── AIGenerator → Claude API with tool calling
    ├── SessionManager → conversation history
    └── ToolManager → executes CourseSearchTool
```

### Query Flow
1. User submits question → FastAPI `/api/query`
2. RAGSystem gets conversation history, calls AIGenerator
3. Claude decides if search is needed via tool calling
4. If yes: CourseSearchTool queries VectorStore → ChromaDB
5. Claude generates final answer from search results
6. Response returned with sources

### Document Ingestion Flow
1. On startup, `app.py` loads files from `../docs` folder
2. DocumentProcessor parses metadata (title, instructor, lessons)
3. Text is chunked (800 chars, 100 overlap) by sentence boundaries
4. Chunks stored in ChromaDB with course/lesson metadata

## Key Configuration (backend/config.py)

- `CHUNK_SIZE`: 800 characters per chunk
- `CHUNK_OVERLAP`: 100 characters overlap
- `MAX_RESULTS`: 5 search results
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## Expected Document Format (in /docs folder)

```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
...
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/query` | POST | Process question, returns `{answer, sources[], session_id}` |
| `/api/courses` | GET | Get course stats `{total_courses, course_titles[]}` |

## Data Models (backend/models.py)

- `Course`: title, course_link, instructor, lessons[]
- `Lesson`: lesson_number, title, lesson_link
- `CourseChunk`: content, course_title, lesson_number, chunk_index

## Environment Variables

Requires `ANTHROPIC_API_KEY` in `.env` file (see `.env.example`).
