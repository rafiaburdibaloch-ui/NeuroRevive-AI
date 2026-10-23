"""Scoring, recommendations, plans and calendar helpers. No Streamlit code here."""
from __future__ import annotations

import random
import statistics
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd

DIFFICULTIES = ["easy", "medium", "hard"]
TASKS = {"memory": "Memory Matrix", "focus": "Focus Finder", "reaction": "Reaction Rush"}
REACTION_ROUNDS = 5

MEMORY_CONFIG = {
    "easy": {"size": 3, "cells": 4, "show": 3.0},
    "medium": {"size": 4, "cells": 6, "show": 3.0},
    "hard": {"size": 5, "cells": 8, "show": 2.5},
}
FOCUS_CONFIG = {
    "easy": {"size": 6, "targets": 6, "target": "Q", "distractors": "O"},
    "medium": {"size": 8, "targets": 8, "target": "E", "distractors": "F"},
    "hard": {"size": 10, "targets": 10, "target": "p", "distractors": "qbd"},
}
PURPOSE = {
    "memory": "Practises holding a short-lived spatial pattern in mind and reproducing it.",
    "focus": "Practises sustained visual attention and ignoring similar-looking distractors.",
    "reaction": "Practises responding steadily to a cue while resisting the urge to click early.",
}
NO_DATA_RULES = ("R1-no-data", "R2-few-data")


# ---------------------------------------------------------------- task generation
def generate_memory(difficulty: str, seed: int) -> set:
    cfg = MEMORY_CONFIG[difficulty]
    rng = random.Random(seed)
    return set(rng.sample(range(cfg["size"] ** 2), cfg["cells"]))


def generate_focus(difficulty: str, seed: int):
    cfg = FOCUS_CONFIG[difficulty]
    rng = random.Random(seed)
    total = cfg["size"] ** 2
    target_idx = set(rng.sample(range(total), cfg["targets"]))
    grid = [
        cfg["target"] if i in target_idx else rng.choice(cfg["distractors"])
        for i in range(total)
    ]
    return grid, target_idx


# ---------------------------------------------------------------- scoring
def score_selection(correct: set, selected: set) -> dict:
    """Accuracy = share of targets found. Score also subtracts false alarms."""
    n = len(correct)
    hits = len(correct & selected)
    false_pos = len(selected - correct)
    misses = len(correct - selected)
    accuracy = 100.0 * hits / n if n else 0.0
    score = 100.0 * max(0, hits - false_pos) / n if n else 0.0
    return {
        "hits": hits, "false_positives": false_pos, "misses": misses, "targets": n,
        "accuracy": round(accuracy, 1), "score": round(score, 1),
    }


def score_reaction(rts: list, false_starts: int) -> dict:
    """Score maps median reaction time (1200 ms -> 0, 300 ms -> 100), minus 10 per false start."""
    accuracy = round(100.0 * len(rts) / REACTION_ROUNDS, 1)
    if not rts:
        return {"median_ms": None, "mean_ms": None, "best_ms": None, "score": 0.0, "accuracy": accuracy}
    median = statistics.median(rts)
    raw = (1200 - median) / 900 * 100 - 10 * false_starts
    return {
        "median_ms": round(median), "mean_ms": round(statistics.mean(rts)),
        "best_ms": min(rts), "score": round(max(0.0, min(100.0, raw)), 1), "accuracy": accuracy,
    }


# ---------------------------------------------------------------- history helpers
def task_rows(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0]
    return df[df["task"] == task].sort_values(["created_at", "id"])


def summarize_tasks(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, name in TASKS.items():
        r = task_rows(df, key)
        if r.empty:
            out.append({"Task": name, "Attempts": 0, "Latest score": None, "Average score": None,
                        "Average accuracy (%)": None, "Last difficulty": "-"})
        else:
            out.append({
                "Task": name, "Attempts": int(len(r)),
                "Latest score": round(float(r.iloc[-1]["score"]), 1),
                "Average score": round(float(r["score"].mean()), 1),
                "Average accuracy (%)": round(float(r["accuracy"].astype(float).mean()), 1),
                "Last difficulty": str(r.iloc[-1]["difficulty"]).title(),
            })
    return pd.DataFrame(out)


def practice_streak(days: set, today: date) -> int:
    cur = today if today in days else today - timedelta(days=1)
    streak = 0
    while cur in days:
        streak += 1
        cur -= timedelta(days=1)
    return streak


def baseline_difficulties(df: pd.DataFrame) -> dict:
    out = {}
    for key in TASKS:
        r = task_rows(df, key)
        if not r.empty:
            out[key] = str(r.iloc[0]["difficulty"])
    return out


def compare_latest(rows: pd.DataFrame):
    """Latest attempt vs the previous one at the SAME difficulty (like-for-like)."""
    if len(rows) < 2:
        return None
    latest = rows.iloc[-1]
    same = rows[rows["difficulty"] == latest["difficulty"]]
    if len(same) < 2:
        return None
    return same.iloc[-2], same.iloc[-1]


# ---------------------------------------------------------------- recommendations
def next_difficulty(current: str, recent_acc: list) -> str:
    idx = DIFFICULTIES.index(current)
    if len(recent_acc) >= 2:
        avg = statistics.mean(recent_acc[-2:])
        if avg >= 85 and idx < len(DIFFICULTIES) - 1:
            idx += 1
        elif avg < 50 and idx > 0:
            idx -= 1
    return DIFFICULTIES[idx]


def _rec(task, difficulty, headline, detail, rule):
    return {"task": task, "difficulty": difficulty, "headline": headline, "detail": detail, "rule": rule}


def recommend_task(df: pd.DataFrame, task: str) -> dict:
    """Transparent, interpretable rules. Rule IDs are shown to the user."""
    name = TASKS[task]
    rows = task_rows(df, task)
    n = len(rows)
    default_diff = "standard" if task == "reaction" else "easy"

    if n == 0:
        return _rec(task, default_diff, f"Complete an initial {name} assessment",
                    "There are no results yet, so nothing can be personalised. Start with one attempt.",
                    "R1-no-data")
    last_diff = str(rows.iloc[-1]["difficulty"])
    if n == 1:
        return _rec(task, last_diff, f"Do a second {name} attempt",
                    "One result is not enough to see a pattern. Repeat at the same level.",
                    "R2-few-data")

    if task == "reaction":
        recent = rows.tail(3)
        avg_fs = statistics.mean(
            [float(d.get("false_starts", 0)) for d in recent["details_obj"]])
        all_rts = [x for d in recent["details_obj"] for x in d.get("rts", [])]
        cv = (statistics.pstdev(all_rts) / statistics.mean(all_rts)) if len(all_rts) >= 6 else None
        if avg_fs >= 2:
            return _rec(task, "standard", "Slow down and wait for the cue",
                        f"You averaged {avg_fs:.1f} false starts per session. Focus on waiting for the "
                        "green cue before clicking.", "R6-false-starts")
        if cv is not None and cv > 0.25:
            return _rec(task, "standard", "Add rounds under consistent conditions",
                        "Your reaction times vary quite a lot between rounds. This is common and may reflect "
                        "device, connection or distraction. Try the same device, a quiet space and a few "
                        "extra sessions before reading anything into it.", "R7-variability")
        return _rec(task, "standard", "Keep a steady routine",
                    "Reaction times and false starts look fairly consistent. Keep practising under the "
                    "same conditions so results stay comparable.", "R8-steady")

    recent_acc = rows[rows["difficulty"] == last_diff]["accuracy"].astype(float).tail(3).tolist()
    new = next_difficulty(last_diff, recent_acc)
    if DIFFICULTIES.index(new) > DIFFICULTIES.index(last_diff):
        return _rec(task, new, f"Try {name} on {new.title()}",
                    f"Your last attempts at {last_diff.title()} averaged "
                    f"{statistics.mean(recent_acc[-2:]):.0f}% accuracy, so a modest step up may suit you.",
                    "R3-step-up")
    if DIFFICULTIES.index(new) < DIFFICULTIES.index(last_diff):
        return _rec(task, new, f"Ease back to {new.title()} for {name}",
                    f"Your last attempts at {last_diff.title()} averaged "
                    f"{statistics.mean(recent_acc[-2:]):.0f}% accuracy. An easier version lets you build "
                    "consistency before moving up.", "R4-step-down")
    return _rec(task, last_diff, f"Keep practising {name} on {last_diff.title()}",
                "Recent accuracy is in a middle range, or there are not yet two attempts at this level. "
                "Repeating builds a like-for-like history.", "R5-hold")


def pick_next_task(df: pd.DataFrame):
    """Untried tasks first; otherwise the task practised least recently (balanced practice)."""
    for key, name in TASKS.items():
        if task_rows(df, key).empty:
            return key, f"You haven't tried {name} yet."
    last = {key: task_rows(df, key)["created_at"].max() for key in TASKS}
    key = min(last, key=last.get)
    return key, "It has been the longest since you practised this one, which keeps practice balanced."


def is_personalised(recs: dict) -> bool:
    return all(r["rule"] not in NO_DATA_RULES for r in recs.values())


def llm_payload(df: pd.DataFrame, recs: dict) -> str:
    """Aggregated numbers only. No participant ID or other identifying information."""
    lines = ["Aggregated task-practice statistics:"]
    for _, r in summarize_tasks(df).iterrows():
        lines.append(f"- {r['Task']}: attempts={r['Attempts']}, latest score={r['Latest score']}, "
                     f"average score={r['Average score']}, average accuracy={r['Average accuracy (%)']}, "
                     f"last difficulty={r['Last difficulty']}")
    lines.append("Rule-based recommendations already computed:")
    for key, rec in recs.items():
        lines.append(f"- {TASKS[key]}: next difficulty={rec['difficulty']} [{rec['rule']}] {rec['headline']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- ML forecast
def ml_forecast(df: pd.DataFrame, task: str) -> dict:
    """Linear regression on score vs attempt number, using only attempts at the latest difficulty."""
    rows = task_rows(df, task)
    if rows.empty:
        return {"ready": False, "n": 0, "difficulty": "-"}
    diff = str(rows.iloc[-1]["difficulty"])
    same = rows[rows["difficulty"] == diff].tail(10)
    n = len(same)
    if n < 4:
        return {"ready": False, "n": n, "difficulty": diff}

    import numpy as np
    from sklearn.linear_model import LinearRegression

    X = np.arange(n).reshape(-1, 1)
    y = same["score"].astype(float).to_numpy()
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])
    pred = float(min(100.0, max(0.0, model.predict([[n]])[0])))
    trend = "rising" if slope > 2 else "falling" if slope < -2 else "flat"
    return {"ready": True, "n": n, "difficulty": diff, "next_pred": pred,
            "slope": slope, "r2": round(float(model.score(X, y)), 2), "trend": trend}


# ---------------------------------------------------------------- training plans
def build_plan(length: int, recs: dict, baseline: dict, personalised: bool) -> dict:
    order = list(TASKS)
    days = []
    for d in range(1, length + 1):
        reassess = d == length
        picks = order if reassess else [order[(d - 1) % 3], order[d % 3]]
        items = []
        for i, task in enumerate(picks):
            diff = baseline.get(task, recs[task]["difficulty"]) if reassess else recs[task]["difficulty"]
            purpose = PURPOSE[task]
            if reassess:
                purpose = ("Reassessment: repeat at the level you first used so the comparison is "
                           "like-for-like. " + purpose)
            items.append({"id": f"{d}-{i}", "task": task, "difficulty": diff,
                          "minutes": 3 if task == "reaction" else 5, "purpose": purpose})
        days.append({"day": d, "title": "Reassessment day" if reassess else "Practice day", "items": items})
    return {"label": "Personalised plan" if personalised else "Starter plan",
            "length": length, "days": days, "done": []}


def next_plan_item(plan: dict):
    done = set(plan.get("done", []))
    for day in plan["days"]:
        for item in day["items"]:
            if item["id"] not in done:
                return day["day"], item
    return None


# ---------------------------------------------------------------- calendar reminder
def make_ics(hhmm: str, days: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    hour, minute = (int(x) for x in hhmm.split(":"))
    start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    end = start + timedelta(minutes=15)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//NeuroRevive AI//Practice reminder//EN",
        "BEGIN:VEVENT", f"UID:{uuid.uuid4()}@neurorevive", f"DTSTAMP:{stamp}",
        f"DTSTART:{start:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}",
        f"RRULE:FREQ=DAILY;COUNT={int(days)}",
        "SUMMARY:NeuroRevive practice session",
        "DESCRIPTION:Short cognitive task practice (prototype - not a medical tool).",
        "BEGIN:VALARM", "ACTION:DISPLAY", "DESCRIPTION:Time for your practice session",
        "TRIGGER:-PT0M", "END:VALARM", "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


# ---------------------------------------------------------------- sample data (always labelled)
def sample_rows(now: datetime | None = None, seed: int = 7) -> list:
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    rows = []
    for k in range(6):
        ts = (now - timedelta(days=5 - k)).isoformat(timespec="seconds")
        for task in ("memory", "focus"):
            base = 45 if task == "memory" else 60
            acc = float(min(100, max(0, base + k * 5 + rng.randint(-6, 6))))
            rows.append(dict(task=task, difficulty="medium", score=acc, accuracy=acc,
                             duration_s=float(rng.randint(15, 40)), details={"sample": True}, created_at=ts))
        fs = rng.choice([0, 0, 1])
        rts = [rng.randint(340, 520) - k * 6 for _ in range(REACTION_ROUNDS - fs)]
        res = score_reaction(rts, fs)
        rows.append(dict(task="reaction", difficulty="standard", score=res["score"], accuracy=res["accuracy"],
                         duration_s=None, created_at=ts,
                         details={"rts": rts, "false_starts": fs, "rounds": REACTION_ROUNDS,
                                  "median_ms": res["median_ms"], "sample": True}))
    return rows
