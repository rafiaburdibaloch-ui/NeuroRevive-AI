"""SQLite persistence for NeuroRevive AI."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path("data") / "neurorevive.db"
COLUMNS = ["id", "participant", "task", "difficulty", "score", "accuracy",
           "duration_s", "details", "is_sample", "created_at"]


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    with _connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant TEXT NOT NULL,
                task TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                score REAL NOT NULL,
                accuracy REAL,
                duration_s REAL,
                details TEXT,
                is_sample INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_results_participant ON results(participant);
            CREATE TABLE IF NOT EXISTS plans (
                participant TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                plan_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prefs (
                participant TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (participant, key)
            );
            """
        )


def save_result(participant, task, difficulty, score, accuracy, duration_s=None,
                details=None, is_sample=False, created_at=None):
    """Returns the new row id, or None if saving failed."""
    try:
        with _connect() as con:
            cur = con.execute(
                "INSERT INTO results (participant, task, difficulty, score, accuracy, duration_s, "
                "details, is_sample, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (participant, task, difficulty, float(score),
                 None if accuracy is None else float(accuracy),
                 None if duration_s is None else float(duration_s),
                 json.dumps(details or {}), 1 if is_sample else 0, created_at or _now()),
            )
            return cur.lastrowid
    except sqlite3.Error:
        return None


def _safe_json(text):
    try:
        obj = json.loads(text) if text else {}
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError):
        return {}


def load_results(participant) -> pd.DataFrame:
    try:
        with _connect() as con:
            df = pd.read_sql_query(
                "SELECT * FROM results WHERE participant = ? ORDER BY created_at, id",
                con, params=(participant,))
    except (sqlite3.Error, pd.errors.DatabaseError):
        df = pd.DataFrame(columns=COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["details_obj"] = df["details"].apply(_safe_json)
    df["median_ms"] = pd.to_numeric(df["details_obj"].apply(lambda d: d.get("median_ms")), errors="coerce")
    df["false_starts"] = pd.to_numeric(df["details_obj"].apply(lambda d: d.get("false_starts")), errors="coerce")
    df["is_sample"] = df["is_sample"].fillna(0).astype(int)
    return df


def load_plan(participant):
    try:
        with _connect() as con:
            row = con.execute("SELECT created_at, plan_json FROM plans WHERE participant = ?",
                              (participant,)).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    plan = _safe_json(row[1])
    if not plan:
        return None
    plan["created_at"] = row[0]
    return plan


def save_plan(participant, plan) -> bool:
    payload = {k: v for k, v in plan.items() if k != "created_at"}
    try:
        with _connect() as con:
            con.execute(
                "INSERT INTO plans (participant, created_at, plan_json) VALUES (?,?,?) "
                "ON CONFLICT(participant) DO UPDATE SET plan_json = excluded.plan_json",
                (participant, _now(), json.dumps(payload)))
        return True
    except sqlite3.Error:
        return False


def delete_plan(participant) -> None:
    try:
        with _connect() as con:
            con.execute("DELETE FROM plans WHERE participant = ?", (participant,))
    except sqlite3.Error:
        pass


def get_pref(participant, key):
    try:
        with _connect() as con:
            row = con.execute("SELECT value FROM prefs WHERE participant = ? AND key = ?",
                              (participant, key)).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def set_pref(participant, key, value) -> bool:
    try:
        with _connect() as con:
            con.execute(
                "INSERT INTO prefs (participant, key, value) VALUES (?,?,?) "
                "ON CONFLICT(participant, key) DO UPDATE SET value = excluded.value",
                (participant, key, value))
        return True
    except sqlite3.Error:
        return False


def delete_sample_results(participant) -> None:
    try:
        with _connect() as con:
            con.execute("DELETE FROM results WHERE participant = ? AND is_sample = 1", (participant,))
    except sqlite3.Error:
        pass


def delete_participant_data(participant) -> None:
    try:
        with _connect() as con:
            for table in ("results", "plans", "prefs"):
                con.execute(f"DELETE FROM {table} WHERE participant = ?", (participant,))
    except sqlite3.Error:
        pass
