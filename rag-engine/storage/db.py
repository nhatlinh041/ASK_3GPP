"""
SQLite connection helpers for chat history persistence.

Single-user dev tool — KHÔNG dùng connection pool, mỗi op mở-đóng kết nối qua
context manager. Driver dùng stdlib `sqlite3` (sync); caller chịu trách nhiệm wrap
qua `asyncio.to_thread` ở async path để không chặn event loop.

Schema được apply idempotent ở `init_db()` lúc FastAPI startup.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# DB file mặc định nằm cạnh code rag-engine/. Override bằng CHAT_DB_PATH env
# (dùng cho test hoặc khi muốn move sang volume mount khác).
DB_PATH = Path(
    os.getenv("CHAT_DB_PATH")
    or Path(__file__).resolve().parent.parent / "data" / "chat.db"
)

# WAL: writer không block reader → SSE response có thể flush msg cuối trong khi
# request khác đọc history. busy_timeout 5s phòng lock contention multi-tab.
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS conversations (
  id          TEXT PRIMARY KEY,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL,
  title       TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content         TEXT NOT NULL,
  thinking        TEXT,
  stages_json     TEXT,
  sources_json    TEXT,
  started_at      INTEGER,
  created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_created
  ON messages(conversation_id, created_at);
"""


def init_db() -> None:
    """Tạo file DB + schema nếu chưa có. Idempotent — gọi mỗi lần startup OK."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Context manager mở-đóng kết nối ngắn cho 1 op CRUD.
    `row_factory = sqlite3.Row` để fetchall/fetchone trả dict-like."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Bật foreign_keys mỗi connection (PRAGMA scope = connection, không persist)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()
