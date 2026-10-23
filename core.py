

# ======================= AI LAYER (ML model + free LLM) =======================
import re

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]


def ml_forecast(df):
    """scikit-learn LinearRegression trained on the user's own attempt history, per task.
    Predicts the next score and classifies the trend. Needs 4+ attempts."""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    out = {}
    for task in TASKS:
        d = df[df["task"] == task].sort_values("created")
        n = len(d)
        if n < 4:
            out[task] = {"n": n, "ready": False}
            continue
        X, y = np.arange(n).reshape(-1, 1), d["score"].values
        m = LinearRegression().fit(X, y)
        slope = float(m.coef_[0])
        out[task] = {"n": n, "ready": True,
                     "next_pred": round(float(np.clip(m.predict([[n]])[0], 0, 100)), 1),
                     "slope": round(slope, 2), "r2": round(float(m.score(X, y)), 2),
                     "trend": "improving" if slope > 1 else "declining" if slope < -1 else "flat"}
    return out


def dropoff_risk(df):
    """Early warning that practice is dropping off (days since last session)."""
    if df.empty:
        return None
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    days = (now - df["created"].max()).days
    return {"days_since_last_practice": days,
            "sessions_last_14_days": int((df["created"] >= now - dt.timedelta(days=14)).sum()),
            "level": "high" if days >= 5 else "medium" if days >= 3 else "low"}


def stats_payload(df, fc, a, risk):
    """Aggregates only. No participant ID is ever sent to the LLM."""
    tasks = {}
    for t in TASKS:
        d = df[df["task"] == t]
        tasks[t] = {"attempts": len(d),
                    "avg_score": round(float(d["score"].mean()), 1) if len(d) else None,
                    "latest_difficulty": str(d["difficulty"].iloc[-1]) if len(d) else None,
                    "ml_forecast": fc.get(t),
                    "rule_recommendation": next(({"difficulty": r["difficulty"], "action": r["action"]}
                                                 for r in a["recs"] if r["task"] == t), None)}
    return {"tasks": tasks, "practice_dropoff_risk": risk}


def llm_json(system, user, api_key):
    """Calls Groq (free tier), trying several models. Returns (parsed_json, model_name)."""
    import requests
    last = None
    for model in GROQ_MODELS:
        try:
            r = requests.post(GROQ_URL, timeout=40,
                              headers={"Authorization": f"Bearer {api_key}",
                                       "Content-Type": "application/json"},
                              json={"model": model, "temperature": 0.3,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", txt, re.S).group(0)), model
        except Exception as e:
            last = e
    raise RuntimeError(f"All models failed: {last}")


SAFETY = ("You are a supportive practice coach inside a student prototype. Use ONLY the data given. "
          "Never diagnose, never comment on intelligence or brain health, remind that short games "
          "are noisy. Return ONLY valid JSON, no markdown.")


def llm_coach(stats, lang, key):
    user = (f"Write in {lang}. Data: {json.dumps(stats)}\n"
            'Return JSON: {"summary": "2-3 sentences", "tips": ["tip1","tip2","tip3"]}')
    data, model = llm_json(SAFETY, user, key)
    if not isinstance(data.get("summary"), str) or not isinstance(data.get("tips"), list):
        raise ValueError("bad coach JSON")
    return data, model


def validate_plan(plan, days):
    """Never trust LLM output: check structure, task names and difficulty before saving."""
    if not isinstance(plan, list) or len(plan) != days:
        raise ValueError("wrong number of days")
    clean = []
    for i, d in enumerate(plan, start=1):
        items = []
        for it in d.get("items", [])[:3]:
            t = str(it.get("task", "")).lower()
            if t not in TASKS:
                raise ValueError("bad task")
            diff = "standard" if t == "reaction" else str(it.get("difficulty", "")).lower()
            if t != "reaction" and diff not in LEVELS:
                raise ValueError("bad difficulty")
            mins = max(2, min(10, int(it.get("minutes", 5))))
            items.append({"task": t, "difficulty": diff, "minutes": mins,
                          "purpose": str(it.get("purpose", PURPOSE[t]))[:180]})
        if not items:
            raise ValueError("empty day")
        clean.append({"day": i, "items": items})
    return clean


def llm_plan(days, stats, lang, key):
    user = (f"Write the 'purpose' text in {lang} (max 15 words each). Data: {json.dumps(stats)}\n"
            f"Create a {days}-day practice plan, 1-3 items per day, 3-8 minutes each. Tasks: memory, "
            "focus, reaction. Difficulty: easy/medium/hard (reaction is always 'standard'). Follow each "
            "task's rule_recommendation difficulty; if ml_forecast is flat, vary the exercises; if "
            "dropoff risk is medium/high, make early days shorter. The final day must reassess all "
            "three tasks at the same difficulty.\n"
            'Return JSON: {"plan":[{"day":1,"items":[{"task":"memory","difficulty":"easy",'
            '"minutes":5,"purpose":"..."}]}]}')
    data, model = llm_json(SAFETY, user, key)
    return validate_plan(data.get("plan"), days), model
