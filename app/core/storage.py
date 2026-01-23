from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = Path("reviews/reviews.db")

def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path.as_posix(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db(db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        well_id TEXT NOT NULL,
        cutoff INTEGER,
        true_ct REAL,
        pred_ct REAL,
        abs_error REAL,
        status TEXT DEFAULT 'Unreviewed',
        tags TEXT DEFAULT '',
        comment TEXT DEFAULT '',
        reviewer TEXT DEFAULT '',
        updated_at TEXT NOT NULL
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ann_key ON annotations(model_id, run_id, well_id, cutoff);")
    conn.commit()
    conn.close()

def upsert_annotation(
    model_id: str,
    run_id: str,
    well_id: str,
    cutoff: Optional[int],
    true_ct: Optional[float],
    pred_ct: Optional[float],
    abs_error: Optional[float],
    status: str,
    tags: List[str],
    comment: str,
    reviewer: str = ""
) -> None:
    init_db()
    conn = _connect()
    now = datetime.utcnow().isoformat(timespec="seconds")

    tags_str = ";".join([t.strip() for t in tags if t.strip()])
    # 간단 upsert: 동일 key가 있으면 최신으로 덮어쓰기
    conn.execute("""
    DELETE FROM annotations
    WHERE model_id=? AND run_id=? AND well_id=? AND (cutoff IS ? OR cutoff=?);
    """, (model_id, run_id, well_id, cutoff, cutoff))
    conn.execute("""
    INSERT INTO annotations (model_id, run_id, well_id, cutoff, true_ct, pred_ct, abs_error, status, tags, comment, reviewer, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (model_id, run_id, well_id, cutoff, true_ct, pred_ct, abs_error, status, tags_str, comment, reviewer, now))
    conn.commit()
    conn.close()

def fetch_annotations(model_id: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    init_db()
    conn = _connect()
    if model_id:
        cur = conn.execute("""
        SELECT model_id, run_id, well_id, cutoff, true_ct, pred_ct, abs_error, status, tags, comment, reviewer, updated_at
        FROM annotations
        WHERE model_id=?
        ORDER BY updated_at DESC
        LIMIT ?;
        """, (model_id, limit))
    else:
        cur = conn.execute("""
        SELECT model_id, run_id, well_id, cutoff, true_ct, pred_ct, abs_error, status, tags, comment, reviewer, updated_at
        FROM annotations
        ORDER BY updated_at DESC
        LIMIT ?;
        """, (limit,))
    rows = cur.fetchall()
    conn.close()

    cols = ["model_id","run_id","well_id","cutoff","true_ct","pred_ct","abs_error","status","tags","comment","reviewer","updated_at"]
    return [dict(zip(cols, r)) for r in rows]
