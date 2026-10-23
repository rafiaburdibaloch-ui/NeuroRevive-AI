"""core.py - data storage, scoring, recommendation engine, plans, calendar.
No Streamlit imports here, so everything can be unit-tested."""
import datetime as dt
import json
import random
import sqlite3
import statistics
import uuid
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).with_name("neurorevive.db")
LEVELS = ["easy", "medium", "hard"]
TASKS = {"memory": "Memory Matrix", "focus": "Focus Finder", "reaction": "Reaction Rush"}
PURPOSE = {
    "memory": "Practises holding and reproducing short digit sequences.",
    "focus": "Practises visual search for a target among similar-looking distractors.",
    "reaction": "Practises steady, consistent responses to a visual cue.",
}
# memory: (digits, seconds per digit)
MEMORY_CFG = {"easy": (4, 1.0), "medium": (6, 0.8), "hard": (8, 0.6)}
# focus: (rows, cols, number of targets, target, distractors)
FOCUS_CFG = {
    "easy": (4, 5, 4, "E", ["X", "M", "O"]),
    "medium": (5, 6, 7, "E", ["F", "L", "B"]),
    "hard": (6, 8, 12, "E", ["F", "L", "B", "P"]),
}


# ---------------------------------------------------------------- database
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS results(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt TEXT UNIQUE, participant TEXT NOT NULL, task TEXT NOT NULL,
            difficulty TEXT NOT NULL, score REAL NOT NULL, accuracy REAL,
            rt_ms REAL, rt_sd REAL, duration_s REAL, created TEXT NOT NULL,
            sample INTEGER DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS plans(
            participant TEXT, days INTEGER, label TEXT, body TEXT,
            PRIMARY KEY(participant, days))""")
        c.execute("""CREATE TABLE IF NOT EXISTS plan_progress(
            participant TEXT, days INTEGER, day INTEGER, item INTEGER,
            PRIMARY KEY(participant, days, day, item))""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings(
            participant TEXT PRIMARY KEY, practice_time TEXT)""")


def _now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def save_result(pid, task, difficulty, score, accuracy=None, rt_ms=None, rt_sd=None,
                duration_s=None, attempt=None, created=None, sample=0):
    """INSERT OR IGNORE on a unique attempt id => one submission can never be saved twice."""
    attempt = attempt or uuid.uuid4().hex
    with _conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO results(attempt,participant,task,difficulty,score,accuracy,"
            "rt_ms,rt_sd,duration_s,created,sample) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (attempt, pid, task, difficulty, float(score), accuracy, rt_ms, rt_sd,
             duration_s, created or _now(), sample))
        return cur.rowcount == 1


COLUMNS = ["id", "attempt", "participant", "task", "difficulty", "score", "accuracy",
           "rt_ms", "rt_sd", "duration_s", "created", "sample"]


def get_results(pid):
    with _conn() as c:
        rows = c.execute("SELECT * FROM results WHERE participant=? ORDER BY created, id",
                         (pid,)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=COLUMNS)
    df["created"] = pd.to_datetime(df["created"])
    df["task_name"] = df["task"].map(TASKS)
    return df


def delete_sample(pid):
    with _conn() as c:
        c.execute("DELETE FROM results WHERE participant=? AND sample=1", (pid,))


def generate_sample(pid):
    """Clearly-labelled demo data (sample=1) so the dashboard is not empty on stage."""
    rng = random.Random(42)
    base = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    plan = [("memory", "easy", 70), ("focus", "easy", 72), ("reaction", "standard", 68),
            ("memory", "easy", 80), ("focus", "easy", 85), ("reaction", "standard", 72),
            ("memory", "medium", 62), ("focus", "medium", 74), ("reaction", "standard", 76)]
    for i, (task, level, score) in enumerate(plan):
        when = (base - dt.timedelta(days=len(plan) - i)).isoformat(timespec="seconds")
        score = min(100, score + rng.randint(-3, 3))
        rt = rng.randint(380, 520) if task == "reaction" else None
        save_result(pid, task, level, score, accuracy=round(score / 100, 2), rt_ms=rt,
                    rt_sd=(rt * 0.2 if rt else None), duration_s=rng.randint(15, 60),
                    created=when, sample=1)


# ----------------------------------------------------------------- scoring
def generate_sequence(level, seed):
    n, _ = MEMORY_CFG[level]
    rng = random.Random(seed)
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def score_memory(target, answer):
    """Position-by-position match. Score 0-100 = % of digits in the correct place."""
    digits = "".join(ch for ch in answer if ch.isdigit())
    correct = sum(1 for a, b in zip(target, digits) if a == b)
    return {"score": round(100 * correct / len(target), 1), "correct": correct, "total": len(target)}


def generate_grid(level, seed):
    rows, cols, k, target, distractors = FOCUS_CFG[level]
    rng = random.Random(seed)
    n = rows * cols
    targets = set(rng.sample(range(n), k))
    cells = [target if i in targets else rng.choice(distractors) for i in range(n)]
    return cells, sorted(targets)


def score_focus(targets, picked):
    """F1 of picked cells vs true targets (penalises both misses and false picks).
    Time is reported separately and does NOT affect the score."""
    t, p = set(targets), set(picked)
    tp, fp, fn = len(t & p), len(p - t), len(t - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"score": round(100 * f1, 1), "accuracy": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def score_reaction(rts, false_starts):
    """Score = 100 - (mean RT - 250ms)/6 - 8 per false start, clamped 0-100."""
    if not rts:
        return {"score": 0.0, "accuracy": 0.0, "mean": None, "sd": None}
    mean = statistics.fmean(rts)
    sd = statistics.pstdev(rts) if len(rts) > 1 else 0.0
    score = max(0.0, min(100.0, 100 - (mean - 250) / 6 - 8 * false_starts))
    return {"score": round(score, 1), "accuracy": round(len(rts) / (len(rts) + false_starts), 3),
            "mean": round(mean, 1), "sd": round(sd, 1)}


# ------------------------------------------------------- recommendation engine
def current_streak(df):
    if df.empty:
        return 0
    days = set(df["created"].dt.date)
    day = dt.datetime.now(dt.timezone.utc).date()
    if day not in days:
        day -= dt.timedelta(days=1)
    streak = 0
    while day in days:
        streak += 1
        day -= dt.timedelta(days=1)
    return streak


def analyse(df):
    """Transparent, testable rules. Returns findings + one recommendation per task."""
    findings, recs = [], []
    for task, name in TASKS.items():
        d = df[df["task"] == task].sort_values("created")
        n = len(d)
        std = "standard" if task == "reaction" else "easy"
        if n < 2:
            recs.append({"task": task, "difficulty": std, "action": "assess",
                         "reason": f"Only {n} {name} attempt(s) saved. Complete at least 2 initial "
                                   f"attempts before personalising."})
            continue
        last = d.tail(3)
        avg = float(last["score"].mean())
        cur = str(last["difficulty"].iloc[-1])
        findings.append(f"{name}: {n} attempts, average {d['score'].mean():.0f}, latest "
                        f"{d['score'].iloc[-1]:.0f} at {cur} level (last-3 average {avg:.0f}).")
        if task == "reaction":
            cvs = (last["rt_sd"] / last["rt_ms"]).dropna()
            cv = float(cvs.mean()) if len(cvs) else 0.0
            if cv > 0.30:
                recs.append({"task": task, "difficulty": std, "action": "consistency",
                             "reason": f"Reaction times varied a lot between rounds (variation "
                                       f"{cv:.0%} of the mean). Practise in the same quiet setting "
                                       f"with the same device; extra rounds help stabilise it. "
                                       f"This is about measurement consistency, not health."})
            elif last["accuracy"].dropna().mean() < 0.8:
                recs.append({"task": task, "difficulty": std, "action": "keep",
                             "reason": "Several false starts. Wait for the green cue before clicking."})
            else:
                recs.append({"task": task, "difficulty": std, "action": "keep",
                             "reason": "Steady results. Keep practising under the same conditions."})
            continue
        idx = LEVELS.index(cur) if cur in LEVELS else 0
        if avg >= 85 and idx < 2:
            recs.append({"task": task, "difficulty": LEVELS[idx + 1], "action": "up",
                         "reason": f"Last-3 average {avg:.0f} at {cur}. Try a modest step up to "
                                   f"{LEVELS[idx + 1]}."})
        elif avg < 55 and idx > 0:
            recs.append({"task": task, "difficulty": LEVELS[idx - 1], "action": "down",
                         "reason": f"Last-3 average {avg:.0f} at {cur}. A shorter/easier version "
                                   f"({LEVELS[idx - 1]}) lets you build consistency first."})
        else:
            recs.append({"task": task, "difficulty": cur, "action": "keep",
                         "reason": f"Last-3 average {avg:.0f} at {cur}. Stay at this level."})
    caveat = ("Scores from a few short games can't measure intelligence or brain health, and early "
              "improvement partly reflects practice effects.")
    return {"findings": findings, "recs": recs, "caveat": caveat, "source": "Rule-based engine"}


def next_activity(df, recs):
    for r in recs:
        if r["action"] == "assess":
            return r
    last_seen = {t: df[df["task"] == t]["created"].max() for t in TASKS}
    oldest = min(last_seen, key=lambda t: last_seen[t])
    return next(r for r in recs if r["task"] == oldest)


def build_plan(days, recs):
    """Returns (plan, label). Starter plan if we lack data for any task."""
    by = {r["task"]: r for r in recs}
    starter = any(r["action"] == "assess" for r in recs)
    order = list(TASKS)

    def item(t, minutes=5, note=None):
        return {"task": t, "difficulty": by[t]["difficulty"], "minutes": minutes,
                "purpose": note or PURPOSE[t]}

    plan = []
    for d in range(1, days + 1):
        if d in (7, days):
            items = [item(t, 4, "Reassessment: repeat at the same difficulty to compare like with like.")
                     for t in order]
        else:
            items = [item(order[(d - 1) % 3]), item(order[d % 3])]
        plan.append({"day": d, "items": items})
    return plan, ("Starter plan" if starter else "Personalised plan")


def save_plan(pid, days, plan, label):
    with _conn() as c:
        c.execute("REPLACE INTO plans VALUES(?,?,?,?)", (pid, days, label, json.dumps(plan)))
        c.execute("DELETE FROM plan_progress WHERE participant=? AND days=?", (pid, days))


def get_plan(pid, days):
    with _conn() as c:
        r = c.execute("SELECT label, body FROM plans WHERE participant=? AND days=?",
                      (pid, days)).fetchone()
    return (json.loads(r["body"]), r["label"]) if r else None


def set_done(pid, days, day, item, done):
    with _conn() as c:
        if done:
            c.execute("INSERT OR IGNORE INTO plan_progress VALUES(?,?,?,?)", (pid, days, day, item))
        else:
            c.execute("DELETE FROM plan_progress WHERE participant=? AND days=? AND day=? AND item=?",
                      (pid, days, day, item))


def get_done(pid, days):
    with _conn() as c:
        rows = c.execute("SELECT day, item FROM plan_progress WHERE participant=? AND days=?",
                         (pid, days)).fetchall()
    return {(r["day"], r["item"]) for r in rows}


def plan_completion(pid, days):
    saved = get_plan(pid, days)
    if not saved:
        return None
    total = sum(len(d["items"]) for d in saved[0])
    return len(get_done(pid, days)), total


# ---------------------------------------------------------------- reminders
def get_time(pid):
    with _conn() as c:
        r = c.execute("SELECT practice_time FROM settings WHERE participant=?", (pid,)).fetchone()
    return r["practice_time"] if r else "18:00"


def set_time(pid, hhmm):
    with _conn() as c:
        c.execute("REPLACE INTO settings VALUES(?,?)", (pid, hhmm))


def make_ics(hhmm, days, activity):
    start = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t = hhmm.replace(":", "") + "00"
    return "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//NeuroRevive AI//EN", "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}@neurorevive", f"DTSTAMP:{stamp}", f"DTSTART:{start}T{t}",
        "DURATION:PT10M", f"RRULE:FREQ=DAILY;COUNT={days}",
        "SUMMARY:NeuroRevive practice session", f"DESCRIPTION:Suggested: {activity}",
        "END:VEVENT", "END:VCALENDAR"])


# ------------------------------------------------------------ optional LLM
def llm_explain(summary_text, api_key, model="claude-sonnet-5-5"):
    """Optional. Sends ONLY aggregate scores (no participant ID). Raises on failure."""
    import requests
    prompt = ("You are a supportive practice coach in a student prototype. Using ONLY the data "
              "below, write 3-4 short, friendly sentences of practice advice. Do not diagnose, "
              "do not mention intelligence or brain health, and note results are from short games.\n\n"
              + summary_text)
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=30,
                      headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": model, "max_tokens": 300,
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])
