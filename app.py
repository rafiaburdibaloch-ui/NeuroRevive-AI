"""NeuroRevive AI - Streamlit entry point. Uses core.py (logic, DB, ML, LLM) and style.css (theme)."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, time as dtime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import core
import games

st.set_page_config(page_title="NeuroRevive AI", page_icon="🧠", layout="wide")

PID_RE = re.compile(r"^[A-Za-z0-9_-]{3,24}$")
PAGES = ["Dashboard", "Memory Matrix", "Focus Finder", "Reaction Rush", "Analysis",
         "Training Plan", "Progress", "Reminders", "About"]
TASK_PAGE = {"memory": "Memory Matrix", "focus": "Focus Finder", "reaction": "Reaction Rush"}
PALETTE = ["#8B5CF6", "#22D3EE", "#6366F1", "#F472B6"]
CSS_PATH = Path(__file__).with_name("style.css")

DISCLAIMER = ("NeuroRevive AI is a prototype for cognitive task practice and self-monitoring. It does not "
              "diagnose any condition, measure intelligence or brain health, or provide clinically validated "
              "rehabilitation. If you have health concerns, please speak to a qualified professional.")


# ------------------------------------------------------------------ helpers
@st.cache_resource
def _init_db():
    core.init_db()
    return True


def load_css():
    try:
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except OSError:
        st.warning("style.css not found; using default theme.")


def go(page: str):
    st.session_state["nav"] = page


def get_secret(name):
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def llm_keys() -> dict:
    keys = {n: get_secret(n) for n in ("GEMINI_API_KEY", "GROQ_API_KEY")}
    return {k: v for k, v in keys.items() if v}


def ensure_pid() -> str:
    if "pid" not in st.session_state:
        qp = st.query_params.get("pid")
        st.session_state["pid"] = qp if qp and PID_RE.match(qp) else "guest-" + uuid.uuid4().hex[:6]
        st.query_params["pid"] = st.session_state["pid"]
    return st.session_state["pid"]


def switch_pid(new_pid: str):
    st.session_state["pid"] = new_pid
    st.query_params["pid"] = new_pid
    for k in ("mem", "foc", "rr", "ai_coach"):
        st.session_state.pop(k, None)
    st.rerun()


def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10), legend_title_text="", font=dict(color="#E6E9F5"))
    return fig


def quick_links(prefix: str):
    for col, (key, name) in zip(st.columns(3), core.TASKS.items()):
        col.button(f"Open {name}", key=f"{prefix}_{key}", on_click=go, args=(TASK_PAGE[key],),
                   use_container_width=True)


def card(badge: str, title: str, body: str, tone: str = ""):
    st.markdown(f'<div class="card"><span class="badge {tone}">{badge}</span><h4>{title}</h4>'
                f"<p>{body}</p></div>", unsafe_allow_html=True)


def recent_table(df, n=6):
    t = df.sort_values(["created", "id"], ascending=False).head(n)
    return pd.DataFrame({
        "When (UTC)": t["created"].dt.strftime("%Y-%m-%d %H:%M"),
        "Task": t["task_name"],
        "Difficulty": t["difficulty"].str.title(),
        "Score": t["score"].round(0),
        "Accuracy %": (t["accuracy"] * 100).round(0),
        "Sample": t["sample"].map({1: "sample", 0: ""}),
    })


def sample_controls(pid, has_sample, prefix):
    st.caption("Sample data is generated, not real performance. It is flagged in the database and can be removed.")
    c1, c2 = st.columns(2)
    if c1.button("Load sample data", key=f"{prefix}_load"):
        core.generate_sample(pid)
        st.rerun()
    if has_sample and c2.button("Remove sample data", key=f"{prefix}_remove"):
        core.delete_sample(pid)
        st.rerun()


def delete_all(pid):
    for table in ("results", "plans", "done", "prefs"):
        core._exec(f"DELETE FROM {table} WHERE participant = ?", (pid,))


def ai_context(df):
    """Aggregated-only payload for the LLM, plus the pieces the UI reuses."""
    analysis = core.analyse(df)
    forecast = core.ml_forecast(df)
    risk = core.dropoff_risk(df)
    return analysis, forecast, risk, core.stats_payload(df, forecast, analysis, risk)


# ------------------------------------------------------------------ pages
def page_dashboard(pid, df):
    st.markdown(
        '<div class="hero"><span class="badge">Prototype · cognitive task practice</span>'
        "<h1>NeuroRevive AI</h1>"
        "<p>Practise short, repeatable cognitive tasks, review how your results change, and follow a "
        "practice routine built from your own task results.</p></div>", unsafe_allow_html=True)

    if df.empty:
        card("Getting started", "No results yet",
             "Complete your first mini-test to see scores, trends and a recommendation. "
             "Start with any of the three tasks below.", "amber")
        quick_links("empty")
        with st.expander("Presenting a demo? Load clearly labelled sample data"):
            sample_controls(pid, False, "dash_empty")
        return

    has_sample = bool(df["sample"].any())
    if has_sample:
        st.warning("This view includes **sample data** (generated, not real performance).")

    latest = df.iloc[-1]
    c = st.columns(4)
    c[0].metric("Completed sessions", len(df))
    c[1].metric("Latest score", f"{latest['score']:.0f}", help=f"Task: {latest['task_name']}")
    c[2].metric("Average score (all tasks)", f"{df['score'].mean():.0f}")
    c[3].metric("Practice streak (UTC days)", core.current_streak(df))

    analysis = core.analyse(df)
    nxt = core.next_activity(df, analysis["recs"])
    risk = core.dropoff_risk(df)
    left, right = st.columns([3, 2])
    with left:
        fig = px.line(df, x="created", y="score", color="task_name", markers=True,
                      color_discrete_sequence=PALETTE, labels={"created": "Date", "score": "Score", "task_name": "Task"})
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption("Scores come from different tasks and difficulty levels. Compare each line only with itself.")
    with right:
        card("Recommended next", f"{core.TASKS[nxt['task']]} · {nxt['difficulty'].title()}", nxt["reason"])
        st.button(f"Start {core.TASKS[nxt['task']]}", type="primary", on_click=go,
                  args=(TASK_PAGE[nxt["task"]],), key="dash_next")
        if risk:
            tone = {"low": "", "medium": "amber", "high": "amber"}[risk["level"]]
            card(f"Routine check · {risk['level']}", "Practice consistency",
                 f"{risk['days_since_last_practice']} day(s) since your last session; "
                 f"{risk['sessions_last_14_days']} session(s) in the last 14 days.", tone)

    st.subheader("Recent activity")
    st.dataframe(recent_table(df), hide_index=True, use_container_width=True)
    st.subheader("Quick links")
    quick_links("dash")


def page_analysis(pid, df):
    st.header("🔍 Performance analysis")
    if df.empty:
        st.info("No saved results yet. Complete at least one attempt of each task to unlock analysis.")
        quick_links("an_empty")
        return
    analysis, forecast, risk, stats = ai_context(df)

    for line in analysis["findings"]:
        st.write("• " + line)
    st.subheader("Recommendations")
    for r in analysis["recs"]:
        card(f"Rule engine · {r['action']}", f"{core.TASKS[r['task']]} · {r['difficulty'].title()}",
             r["reason"], "violet")
    if any(r["action"] == "assess" for r in analysis["recs"]):
        st.info("Some tasks have fewer than two attempts, so recommendations there are limited.")

    st.subheader("ML forecast (scikit-learn)")
    for col, (key, name) in zip(st.columns(3), core.TASKS.items()):
        f = forecast[key]
        if f["ready"]:
            col.metric(name, f"{f['next_pred']:.0f} predicted next",
                       f"{f['trend']} ({f['slope']:+.1f} per attempt)", delta_color="off")
            col.caption(f"Linear regression on your last {f['n']} attempts at one level. R² = {f['r2']}")
        else:
            col.metric(name, "—", f"needs 4+ attempts at one level ({f['n']} so far)", delta_color="off")
    st.caption("The model learns only from your own few results, so treat it as a rough practice trend, "
               "not a prediction about ability or health.")

    st.subheader("AI coach (optional)")
    keys = llm_keys()
    lang = st.session_state.get("lang", "English")
    if not keys:
        st.info("No GEMINI_API_KEY or GROQ_API_KEY is configured, so only the transparent rules above are used. "
                "No language model was called.")
    else:
        st.caption(f"Only aggregated numbers are sent to the model (reply language: {lang}). "
                   "No participant ID is included.")
        if st.button("Generate AI coach feedback", key="ai_btn"):
            with st.spinner("Asking the model…"):
                try:
                    st.session_state["ai_coach"] = core.llm_coach(stats, lang, keys)
                except Exception as exc:
                    st.warning(f"The AI call failed ({exc}). The rule-based results above still apply.")
        if st.session_state.get("ai_coach"):
            result, src = st.session_state["ai_coach"]
            st.markdown(f'<span class="badge violet">AI-generated · {src}</span>', unsafe_allow_html=True)
            st.write(result["summary"])
            for tip in result["tips"]:
                st.markdown(f"- {tip}")
    st.caption(analysis["caveat"])


def _toggle(pid, length, day, idx, wkey):
    core.set_done(pid, length, day, idx, bool(st.session_state.get(wkey)))


def _generate_plan(pid, length, df, use_ai):
    analysis, _, _, stats = ai_context(df)
    try:
        if use_ai:
            plan, src = core.llm_plan(length, stats, st.session_state.get("lang", "English"), llm_keys())
            label = f"AI plan ({src})"
        else:
            plan, label = core.build_plan(length, analysis["recs"])
        core.save_plan(pid, length, plan, label)
    except Exception as exc:
        st.error(f"Could not create the plan: {exc}")
        return
    for k in [k for k in st.session_state if str(k).startswith(f"plan_{pid}_{length}_")]:
        del st.session_state[k]
    st.rerun()


def _plan_buttons(pid, length, df, prefix):
    c1, c2 = st.columns(2)
    if c1.button("Generate rule-based plan", type="primary", key=f"{prefix}_rule"):
        _generate_plan(pid, length, df, False)
    if llm_keys():
        if c2.button("Generate AI plan", key=f"{prefix}_ai"):
            with st.spinner("Asking the model…"):
                _generate_plan(pid, length, df, True)
    else:
        c2.caption("Add GEMINI_API_KEY or GROQ_API_KEY to enable AI-written plans.")


def page_plan(pid, df):
    st.header("🗓️ Training plan")
    length = st.radio("Plan length", [7, 14], horizontal=True, format_func=lambda d: f"{d}-day")
    saved = core.get_plan(pid, length)
    if saved is None:
        recs = core.analyse(df)["recs"]
        if any(r["action"] == "assess" for r in recs):
            st.info("You don't have enough results for every task yet, so a rule-based plan will be a "
                    "**starter plan**. Regenerate it after more attempts.")
        _plan_buttons(pid, length, df, "plan_new")
        return

    plan, label = saved
    done = core.get_done(pid, length)
    total = sum(len(d["items"]) for d in plan)
    st.markdown(f'<span class="badge violet">{label} · {length} days</span>', unsafe_allow_html=True)
    st.progress(len(done) / total if total else 0.0, text=f"{len(done)} of {total} exercises completed")

    nxt = next(((d["day"], it) for d in plan for i, it in enumerate(d["items"]) if (d["day"], i) not in done), None)
    if nxt:
        st.info(f"Next up: Day {nxt[0]} · {core.TASKS[nxt[1]['task']]} ({nxt[1]['difficulty'].title()})")
        st.button(f"Open {core.TASKS[nxt[1]['task']]}", on_click=go, args=(TASK_PAGE[nxt[1]["task"]],),
                  key="plan_open")
    else:
        st.success("Plan complete. Re-take each task and compare like-for-like on the Progress page.")

    for day in plan:
        n_done = sum(1 for i in range(len(day["items"])) if (day["day"], i) in done)
        with st.expander(f"Day {day['day']} · {n_done}/{len(day['items'])} done", expanded=nxt and nxt[0] == day["day"]):
            for i, it in enumerate(day["items"]):
                wkey = f"plan_{pid}_{length}_{day['day']}_{i}"
                st.checkbox(f"{core.TASKS[it['task']]} · {it['difficulty'].title()} · {it['minutes']} min",
                            value=(day["day"], i) in done, key=wkey, on_change=_toggle,
                            args=(pid, length, day["day"], i, wkey))
                st.caption(it["purpose"])
    st.caption("Tick exercises yourself after completing them. Completion is self-reported.")
    with st.expander("Replace this plan (resets ticks)"):
        _plan_buttons(pid, length, df, "plan_re")


def page_progress(pid, df):
    st.header("📈 Progress")
    if df.empty:
        st.info("No results yet. Complete a task and your history will appear here.")
        quick_links("pr_empty")
        with st.expander("Presenting a demo? Load clearly labelled sample data"):
            sample_controls(pid, False, "pr_empty")
        return
    if df["sample"].any():
        st.warning("This view includes **sample data**.")

    comp = [core.plan_completion(pid, d) for d in (7, 14)]
    comp = next((c for c in comp if c), None)
    c = st.columns(3)
    c[0].metric("Completed sessions", len(df))
    c[1].metric("Plan completion", f"{comp[0]}/{comp[1]}" if comp else "No plan")
    c[2].metric("Tasks tried", df["task"].nunique())

    task = st.selectbox("Task", list(core.TASKS), format_func=core.TASKS.get)
    rows = df[df["task"] == task].copy()
    if rows.empty:
        st.info(f"No {core.TASKS[task]} results yet.")
    else:
        rows["Attempt"] = range(1, len(rows) + 1)
        rows["Difficulty"] = rows["difficulty"].str.title()
        rows["Accuracy (%)"] = rows["accuracy"] * 100
        l, r = st.columns(2)
        fig = px.line(rows, x="Attempt", y="score", color="Difficulty", markers=True,
                      color_discrete_sequence=PALETTE, title="Score by attempt")
        fig.update_yaxes(range=[0, 105])
        l.plotly_chart(style_fig(fig), use_container_width=True)
        ycol, ytitle = ("rt_ms", "Mean reaction (ms), lower is faster") if task == "reaction" \
            else ("Accuracy (%)", "Accuracy (%)")
        r.plotly_chart(style_fig(px.line(rows, x="Attempt", y=ycol, color="Difficulty", markers=True,
                                         color_discrete_sequence=PALETTE, title=ytitle)),
                       use_container_width=True)

        same = rows[rows["difficulty"] == rows.iloc[-1]["difficulty"]]
        if len(same) >= 2:
            prev, last = same.iloc[-2], same.iloc[-1]
            st.subheader("Previous vs latest (same difficulty)")
            m = st.columns(2)
            m[0].metric("Score", f"{last['score']:.0f}", f"{last['score'] - prev['score']:+.0f}")
            if task == "reaction" and pd.notna(last["rt_ms"]) and pd.notna(prev["rt_ms"]):
                m[1].metric("Mean reaction (ms)", f"{last['rt_ms']:.0f}", f"{last['rt_ms'] - prev['rt_ms']:+.0f}",
                            delta_color="inverse")
            elif task != "reaction":
                m[1].metric("Accuracy", f"{last['accuracy'] * 100:.0f}%",
                            f"{(last['accuracy'] - prev['accuracy']) * 100:+.0f}")
        else:
            st.caption("Two attempts at the same difficulty are needed for a like-for-like comparison.")

    st.info("Higher scores on a task mean better performance **on that task**. They are not proof of overall "
            "cognitive improvement. Repeating a task causes practice effects, and results vary with device, "
            "mood, sleep and distraction. Compare like-for-like only.")

    with st.expander("Data tools"):
        export = df[["created", "task", "difficulty", "score", "accuracy", "rt_ms", "rt_sd", "duration_s", "sample"]]
        st.download_button("Download my results (CSV)", export.to_csv(index=False).encode("utf-8"),
                           "neurorevive_results.csv", "text/csv")
        sample_controls(pid, bool(df["sample"].any()), "pr")
        confirm = st.checkbox("I understand this permanently deletes all data for this participant ID",
                              key="del_confirm")
        if st.button("Delete all my data", disabled=not confirm, key="del_btn"):
            delete_all(pid)
            st.rerun()


def page_reminders(pid, df):
    st.header("⏰ Reminders")
    try:
        h, m = (int(x) for x in core.get_time(pid).split(":"))
    except ValueError:
        h, m = 18, 0
    chosen = st.time_input("Preferred daily practice time", value=dtime(h, m), step=900)
    hhmm = chosen.strftime("%H:%M")
    if st.button("Save preferred time", key="rem_save"):
        core.set_time(pid, hhmm)
        st.success("Saved.")

    nxt = core.next_activity(df, core.analyse(df)["recs"])
    activity = f"{core.TASKS[nxt['task']]} ({nxt['difficulty'].title()})"
    card("Next recommended session", f"Daily at {hhmm}", activity)

    days = st.radio("Repeat for", [7, 14], horizontal=True, format_func=lambda d: f"{d} days")
    st.download_button("Download calendar reminder (.ics)", core.make_ics(hhmm, days, activity),
                       "neurorevive-reminder.ics", "text/calendar")
    st.caption("The .ics file adds a repeating daily event to your own calendar app. "
               "This web app does not send push notifications or emails.")


def page_about():
    st.header("ℹ️ About & how it works")
    st.write(DISCLAIMER)
    with st.expander("How the tests and scores work", expanded=True):
        st.markdown(
            "- **Memory Matrix:** digits flash one at a time; type them back in order. "
            "Score = % of digits in the correct position.\n"
            "- **Focus Finder:** tick every target among look-alike distractors. Score = F1 of your ticks vs the "
            "true targets (false alarms and misses both count). Completion time is recorded but not scored. "
            "Each grid has a seed so it can be reproduced.\n"
            "- **Reaction Rush:** 5 rounds. Score = 100 − (mean ms − 250) ÷ 6 − 8 per false start. Timing runs "
            "through the server and includes network delay.")
    with st.expander("Where the AI is"):
        st.markdown(
            "1. **Machine learning:** a scikit-learn linear regression on your own recent scores at one "
            "difficulty (minimum 4 attempts) forecasts your next score and trend.\n"
            "2. **Rule engine:** transparent rules choose the next difficulty and activity.\n"
            "3. **Optional language models:** if `GEMINI_API_KEY` and/or `GROQ_API_KEY` are set, a model writes "
            "coach feedback or a practice plan (English or Urdu). It only receives aggregated statistics, and "
            "plan output is validated before use.")
    with st.expander("What data is stored"):
        st.write("An anonymous participant ID, task, difficulty, score, accuracy, timings, your plans and tick "
                 "marks, and your reminder time. There is no password: anyone who knows an ID can view that "
                 "ID's results, so don't use your real name. Export or delete your data on the Progress page.")
    with st.expander("Limitations"):
        st.markdown(
            "- A handful of game scores is noisy and affected by practice, mood, sleep, device and distraction.\n"
            "- Getting better at a game is not evidence of overall cognitive change.\n"
            "- Browser reaction timing has latency; it is not laboratory-grade.\n"
            "- Free hosting may reset stored data. Export your CSV if it matters.\n"
            "- This is not a medical device and not a diagnosis.")


# ------------------------------------------------------------------ main
def main():
    load_css()
    try:
        _init_db()
    except Exception as exc:
        st.error(f"Could not initialise the database: {exc}")
        st.stop()

    pid = ensure_pid()
    st.session_state.setdefault("nav", "Dashboard")

    with st.sidebar:
        st.markdown("## 🧠 NeuroRevive AI")
        st.caption("Prototype · cognitive task practice")
        st.radio("Navigate", PAGES, key="nav")
        st.selectbox("AI language", list(core.LANG), key="lang")
        st.divider()
        with st.form("pid_form"):
            new = st.text_input("Participant ID (anonymous)", value=pid, max_chars=24,
                                help="3-24 letters, numbers, - or _. Don't use your real name.")
            ok = st.form_submit_button("Use this ID")
        if ok:
            new = new.strip()
            if PID_RE.match(new):
                if new != pid:
                    switch_pid(new)
            else:
                st.error("Use 3-24 letters, numbers, - or _.")
        st.caption("Bookmark this page's URL to return to your data. Demo mode: no password.")

    df = core.get_results(pid)
    page = st.session_state["nav"]
    if page == "Dashboard":
        page_dashboard(pid, df)
    elif page == "Memory Matrix":
        games.memory_page(pid)
    elif page == "Focus Finder":
        games.focus_page(pid)
    elif page == "Reaction Rush":
        games.reaction_page(pid)
    elif page == "Analysis":
        page_analysis(pid, df)
    elif page == "Training Plan":
        page_plan(pid, df)
    elif page == "Progress":
        page_progress(pid, df)
    elif page == "Reminders":
        page_reminders(pid, df)
    else:
        page_about()
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


main()
