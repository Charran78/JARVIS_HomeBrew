import sqlite3
import threading
from typing import Dict, Optional

class PreferencesStore:
    def __init__(self, db_path: str = "app.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT)"
            )
            self.conn.commit()
    def set_many(self, prefs: Dict[str, str]):
        with self.lock:
            cur = self.conn.cursor()
            for k, v in prefs.items():
                cur.execute(
                    "INSERT INTO preferences(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, str(v))
                )
            self.conn.commit()
    def get_all(self) -> Dict[str, str]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT key, value FROM preferences")
            rows = cur.fetchall()
        return {k: v for k, v in rows}
    def get(self, key: str) -> Optional[str]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT value FROM preferences WHERE key=?", (key,))
            row = cur.fetchone()
        return row[0] if row else None
