import sqlite3
import threading
from typing import List, Dict, Optional

class HistoryStore:
    def __init__(self, db_path: str = "history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at REAL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts REAL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS citations (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER, text TEXT, score REAL, metadata TEXT)"
            )
            self.conn.commit()
    def create_session(self, session_id: str, created_at: float):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at) VALUES(?, ?)", (session_id, created_at)
            )
            self.conn.commit()
    def append_message(self, session_id: str, role: str, content: str, ts: float):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES(?, ?, ?, ?)",
                (session_id, role, content, ts)
            )
            self.conn.commit()
    def append_message_return_id(self, session_id: str, role: str, content: str, ts: float) -> int:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES(?, ?, ?, ?)",
                (session_id, role, content, ts)
            )
            mid = cur.lastrowid
            self.conn.commit()
            return mid
    def add_citations(self, message_id: int, citations: List[Dict]):
        if not citations:
            return
        with self.lock:
            cur = self.conn.cursor()
            for c in citations:
                text = c.get("text", "")
                score = c.get("score", None)
                metadata = c.get("metadata", {})
                cur.execute(
                    "INSERT INTO citations(message_id, text, score, metadata) VALUES(?, ?, ?, ?)",
                    (message_id, text, score if score is not None else None, str(metadata))
                )
            self.conn.commit()
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
                (session_id,)
            )
            rows = cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
    def get_messages(self, session_id: str) -> List[Dict]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, role, content, ts FROM messages WHERE session_id=? ORDER BY id ASC",
                (session_id,)
            )
            rows = cur.fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2], "ts": r[3]} for r in rows]
    def get_citations_for_message(self, message_id: int) -> List[Dict]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT text, score, metadata FROM citations WHERE message_id=? ORDER BY id ASC",
                (message_id,)
            )
            rows = cur.fetchall()
        res = []
        for t, s, m in rows:
            res.append({"text": t, "score": s, "metadata": m})
        return res
