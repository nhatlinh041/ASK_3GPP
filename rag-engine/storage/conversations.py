"""
ConversationStore — CRUD cho chat history.

Tất cả method đồng bộ (sqlite3 stdlib). Async caller phải wrap qua
`asyncio.to_thread(...)` để không chặn event loop của FastAPI.

Stages lưu RAW JSON (không compact server-side) — single-user dev tool, vài MB
per turn vẫn fit thoải mái trong SQLite TEXT column. Trade-off: simpler hơn,
không phải port `compactStagesForStorage` từ TS sang Python.
"""
import json
import time
import uuid
from typing import Any

from .db import connect


def _now_ms() -> int:
    """Epoch milliseconds — khớp với `Date.now()` ở frontend."""
    return int(time.time() * 1000)


class ConversationStore:
    def create(self) -> str:
        """Tạo conversation mới với UUIDv4 id, trả id."""
        cid = str(uuid.uuid4())
        now = _now_ms()
        with connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, updated_at, title) "
                "VALUES (?, ?, ?, NULL)",
                (cid, now, now),
            )
            conn.commit()
        return cid

    def exists(self, cid: str) -> bool:
        """Check id có trong DB không — dùng để validate request body."""
        with connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (cid,)
            ).fetchone()
        return row is not None

    def get(self, cid: str) -> dict | None:
        """Trả conversation metadata + toàn bộ messages.
        Return None nếu cid không tồn tại."""
        with connect() as conn:
            conv = conn.execute(
                "SELECT id, created_at, updated_at, title FROM conversations WHERE id = ?",
                (cid,),
            ).fetchone()
            if conv is None:
                return None
            rows = conn.execute(
                "SELECT role, content, thinking, stages_json, sources_json, started_at "
                "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
                (cid,),
            ).fetchall()
        # Parse JSON ở Python để frontend nhận shape đúng (stages: array | null)
        messages = [
            {
                "role": r["role"],
                "content": r["content"],
                "thinking": r["thinking"],
                "stages": json.loads(r["stages_json"]) if r["stages_json"] else None,
                "sources": json.loads(r["sources_json"]) if r["sources_json"] else None,
                "startedAt": r["started_at"],
            }
            for r in rows
        ]
        return {
            "id": conv["id"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "title": conv["title"],
            "messages": messages,
        }

    def append_user(self, cid: str, content: str, started_at: int) -> None:
        """Lưu user message NGAY khi nhận query — refresh giữa stream vẫn còn."""
        now = _now_ms()
        with connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, started_at, created_at) "
                "VALUES (?, 'user', ?, ?, ?)",
                (cid, content, started_at, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid)
            )
            # Auto-set title bằng prompt đầu (cắt 80 ký tự) — tiện sau làm sidebar
            conn.execute(
                "UPDATE conversations SET title = ? "
                "WHERE id = ? AND title IS NULL",
                (content[:80], cid),
            )
            conn.commit()

    def append_assistant(
        self,
        cid: str,
        content: str,
        thinking: str | None,
        stages: list[dict[str, Any]],
        sources: list[dict[str, Any]] | None,
        started_at: int,
    ) -> None:
        """Lưu assistant turn sau khi stream xong. stages = raw event list."""
        now = _now_ms()
        with connect() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, thinking, stages_json, sources_json, "
                " started_at, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    content,
                    thinking,
                    json.dumps(stages) if stages else None,
                    json.dumps(sources) if sources else None,
                    started_at,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid)
            )
            conn.commit()

    def delete(self, cid: str) -> bool:
        """ON DELETE CASCADE tự xoá messages. Trả True nếu cid tồn tại trước đó."""
        with connect() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
            conn.commit()
            return cur.rowcount > 0
