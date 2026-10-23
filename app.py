"""NeuroRevive AI - Streamlit entry point."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, time as dtime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

import db
import games
import llm
import logic

st.set_page_config(page_title="NeuroRevive AI", page_icon="🧠", layout="wide")

PID_RE = re.compile(r"^[A-Za-z0-9_-]{3,24}$")
PAGES = ["Dashboard", "Memory Matrix", "Focus Finder", "Reaction Rush", "Analysis",
         "Training Plan", "Progress", "Reminders", "About"]
TASK_PAGE = {"memory": "Memory Matrix", "focus": "Focus Finder", "reaction": "Reaction Rush"}
PALETTE = ["#8B5CF6", "#22D3EE", "#6366F1", "#F472B6"]

CSS = """
:root{--card:#111B3A;--border:#26325F;--cyan:#22D3EE;--muted:#9AA6D1;}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 10% -10%,#1B2559 0%,#0A1030 55%,#070B22 100%);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:#0B1233;border-right:1px solid var(--border);}
h1,h2,h3{letter-spacing:-0.02em;}
.nr-hero{background:linear-gradient(135deg,rgba(99,102,241,.25),rgba(34,211,238,.12));border:1px solid var(--border);
  border-radius:20px;padding:1.6rem 1.8rem;margin-bottom:1rem;}
.nr-hero h1{margin:.4rem 0 .3rem;font-size:2.3rem;
  background:linear-gradient(90deg,#C4B5FD,#67E8F9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nr-hero p{color:var(--muted);margin:0;max-width:760px;}
.nr-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1rem 1.25rem;margin-bottom:.8rem;}
.nr-card h3{margin-top:.4rem;}
.nr-pill{display:inline-block;padding:.12rem .65rem;border-radius:999px;font-size:.75rem;color:var(--cyan);
  border:1px solid var(--border);background:rgba(34,211,238,.08);}
.nr-pill.warn{color:#FBBF24;background:rgba(251,191,36,.08);}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:.8rem 1rem;}
.stButton>button,.stDownloadButton>button{border-radius:12px;}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#6366F1,#8B5CF6);border:none;color:white;}
:focus-visible{outline:3px solid #22D3EE !important;outline-offset:2px;}
.nr-grid{display:grid;gap:8px;max-width:420px;margin:.5rem 0 1rem;}
.nr-cell{aspect-ratio:1/1;border-radius:10px;background:#141E45;border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1.2rem;}
.nr-cell.on{background:linear-gradient(135deg,#22D3EE,#6366F1);border-color:#22D3EE;}
.nr-cell.hit{background:#065F46;border-color:#34D399;}
.nr-cell.miss{background:#78350F;border-color:#FBBF24;}
.nr-cell.fp{background:#7F1D1D;border-color:#F87171;}
.rr-box{height:180px;border-radius:18px;display:flex;align-items:center;justify-content:center;
  font-size:1.6rem;font-weight:700;margin-bottom:.6rem;}
.rr-wait{background:#78350F;color:#FDE68A;border:2px solid #FBBF24;}
.rr-go{background:#065F46;color:#D1FAE5;border:2px solid #34D399;}
"""

DISCLAIMER = ("NeuroRevive AI is a **prototype for cognitive task practice and self-monitoring**. It does not "
              "diagnose any condition, measure intelligence or brain health, or provide clinically validated "
              "rehabilitation. If you have health concerns, please speak to a qualified professional.")


# ------------------------------------------------------------------ helpers
@st.cache_resource
def _init_db():
    db.init_db()
    return True


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


def ensure_pid() -> str:
    if "pid" not in st.session_state:
        qp = st.query_params.get("pid")
        if qp and PID_RE.match(qp):
            st.session_state["pid"] = qp
        else:
            st.session_state["pid"] = "guest-" + uuid.uuid4().hex[:6]
        st.query_params["pid"] = st.session_state["pid"]
    return st.session_state["pid"]


def switch_pid(new_pid: str):
    st.session_state["pid"] = new_pid
    st.query_params["pid"] = new_pid
    for k in ("mem", "foc", "rr", "ai_text"):
        st.session_state.pop(k, None)
    st.rerun()


def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      font=dict(color="#E6E9F5"))
    return fig


def quick_links(prefix: str):
    cols = st.columns(3)
    for col, (key, name) in zip(cols, logic.TASKS.items()):
        col.button(f"Open {name}", key=f"{prefix}_{key}", on_click=go, args=(TASK_PAGE[key],),
                   use_container_width=True)


def recent_table(df, n=6):
    t = df.sort_values(["created_at", "id"], ascending=False).head(n)
    return pd.DataFrame({
        "When (UTC)": t["created_at"].dt.strftime("%Y-%m-%d %H:%M"),
        "Task": t["task"].map(logic.TASKS),
        "Difficulty": t["difficulty"].str.title(),
        "Score": t["score"].astype(float).round(0),
        "Accuracy %": t["accuracy"].astype(float).round(0),
        "Sample": t["is_sample"].map({1: "sample", 0: ""}),
    })


def sample_controls(pid, has_sample, prefix):
    st.caption("Sample data is generated, not real performance. It is flagged in the database and can be removed.")
    c1, c2 = st.columns(2)
    if c1.button("Load sample data", key=f"{prefix}_load"):
        for row in logic.sample_rows():
            db.save_result(pid, is_sample=True, **row)
        st.rerun()
    if has_sample and c2.button("Remove sample data", key=f"{prefix}_remove"):
        db.delete_sample_results(pid)
        st.rerun()


# ------------------------------------------------------------------ pages
def page_dashboard(pid, df):
    st.markdown(
        '<div class="nr-hero"><span class="nr-pill">Prototype · cognitive task practice</span>'
        "<h1>NeuroRevive AI</h1>"
        "<p>Practise short, repeatable cognitive tasks, review how your results change, and follow a "
        "personalised practice routine built from your own task results.</p></div>",
        unsafe_allow_html=True)
    st.caption(DISCLAIMER)

    if df.empty:
        st.markdown(
            '<div class="nr-card"><span class="nr-pill">Getting started</span><h3>No results yet</h3>'
            "<p>Complete your first mini-test to see scores, trends and a personalised recommendation. "
            "Start with any of the three tasks below.</p></div>", unsafe_allow_html=True)
        quick_links("empty")
        with st.expander("Presenting a demo? Load clearly labelled sample data"):
            sample_controls(pid, False, "dash_empty")
        return

    has_sample = bool(df["is_sample"].any())
    if has_sample:
        st.warning("This view includes **sample data** (generated, not real performance).")

    latest = df.sort_values(["created_at", "id"]).iloc[-1]
    streak = logic.practice_streak(set(df["created_at"].dt.date), datetime.now(timezone.utc).date())
    c = st.columns(4)
    c[0].metric("Completed sessions", len(df))
    c[1].metric("Latest score", f"{latest['score']:.0f}", help=f"Task: {logic.TASKS[latest['task']]}")
    c[2].metric("Average score (all tasks)", f"{df['score'].astype(float).mean():.0f}")
    c[3].metric("Practice streak (UTC days)", f"{streak}")

    t, why = logic.pick_next_task(df)
    rec = logic.recommend_task(df, t)
    left, right = st.columns([3, 2])
    with left:
        plot = df.assign(Task=df["task"].map(logic.TASKS))
        fig = px.line(plot, x="created_at", y="score", color="Task", markers=True,
                      color_discrete_sequence=PALETTE, labels={"created_at": "Date", "score": "Score"})
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption("Scores come from different tasks and difficulty levels. Compare each line only with itself.")
    with right:
        st.markdown(
            f'<div class="nr-card"><span class="nr-pill">Recommended next</span>'
            f"<h3>{logic.TASKS[t]} · {rec['difficulty'].title()}</h3><p>{why}</p><p>{rec['detail']}</p></div>",
            unsafe_allow_html=True)
        st.button(f"Start {logic.TASKS[t]}", type="primary", on_click=go, args=(TASK_PAGE[t],), key="dash_next")

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
    recs = {k: logic.recommend_task(df, k) for k in logic.TASKS}
    st.dataframe(logic.summarize_tasks(df), hide_index=True, use_container_width=True)
    if not logic.is_personalised(recs):
        st.info("Some tasks have fewer than two attempts, so recommendations there are limited. "
                "Complete the suggested initial assessments before relying on a personalised plan.")

    st.subheader("ML forecast (scikit-learn)")
    fcols = st.columns(3)
    for col, (key, name) in zip(fcols, logic.TASKS.items()):
        f = logic.ml_forecast(df, key)
        if f["ready"]:
            col.metric(name, f"{f['next_pred']:.0f} predicted next",
                       f"{f['trend']} ({f['slope']:+.1f} per attempt)", delta_color="off")
            col.caption(f"Linear regression on your last {f['n']} attempts at "
                        f"{f['difficulty'].title()}. R² = {f['r2']}")
        else:
            col.metric(name, "—", f"needs 4+ attempts at one level ({f['n']} so far)",
                       delta_color="off")
    st.caption("This model learns only from your own few results, so treat the forecast as a rough "
               "practice trend. It is not a prediction about ability or health.")

    st.subheader("Recommendations")
    for k, rec in recs.items():
        st.markdown(
            f'<div class="nr-card"><span class="nr-pill">Rule-based · {rec["rule"]}</span>'
            f"<h3>{rec['headline']}</h3><p>{rec['detail']}</p>"
            f"<p><strong>Suggested level:</strong> {rec['difficulty'].title()}</p></div>",
            unsafe_allow_html=True)

    st.subheader("AI explanation (optional)")
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.info("No API key is configured, so the recommendations above come from transparent rules only. "
                "No language model was called.")
    else:
        st.caption("Only the aggregated numbers above are sent to the model. No participant ID is included.")
        if st.button("Generate AI explanation", key="ai_btn"):
            with st.spinner("Asking the model…"):
                try:
                    model = get_secret("LLM_MODEL") or llm.DEFAULT_MODEL
                    st.session_state["ai_text"] = (llm.explain(logic.llm_payload(df, recs), api_key, model), model)
                except Exception as exc:
                    st.warning(f"The AI call failed ({type(exc).__name__}). The rule-based results above still apply.")
        if st.session_state.get("ai_text"):
            text, model = st.session_state["ai_text"]
            st.markdown(f'<span class="nr-pill">AI-generated · {model}</span>', unsafe_allow_html=True)
            st.write(text)
    st.caption("A few game scores are noisy and affected by practice, mood, device and distraction. "
               "Nothing here is a statement about intelligence, brain health or any medical condition.")


def _toggle_item(pid, item_id, widget_key):
    plan = db.load_plan(pid)
    if plan is None:
        return
    done = set(plan.get("done", []))
    if st.session_state.get(widget_key):
        done.add(item_id)
    else:
        done.discard(item_id)
    plan["done"] = sorted(done)
    db.save_plan(pid, plan)


def page_plan(pid, df):
    st.header("🗓️ Training plan")
    plan = db.load_plan(pid)
    if plan is None:
        length = st.radio("Plan length", [7, 14], horizontal=True, format_func=lambda d: f"{d}-day")
        recs = {k: logic.recommend_task(df, k) for k in logic.TASKS}
        personalised = logic.is_personalised(recs)
        if personalised:
            st.success("Your plan will use your recent results to choose each task's difficulty.")
        else:
            st.info("You don't have enough results for every task yet, so this will be a **starter plan** "
                    "(easy levels where there is no data). Regenerate it after more attempts.")
        if st.button("Generate plan", type="primary", key="plan_gen"):
            new = logic.build_plan(length, recs, logic.baseline_difficulties(df), personalised)
            if db.save_plan(pid, new):
                st.rerun()
            else:
                st.error("Could not save the plan.")
        return

    created = datetime.fromisoformat(plan["created_at"])
    today_n = max(1, min(plan["length"], (datetime.now(timezone.utc).date() - created.date()).days + 1))
    total = sum(len(d["items"]) for d in plan["days"])
    done = set(plan.get("done", []))
    st.markdown(f'<span class="nr-pill">{plan["label"]} · {plan["length"]} days</span>', unsafe_allow_html=True)
    st.progress(len(done) / total, text=f"{len(done)} of {total} exercises completed")
    nxt = logic.next_plan_item(plan)
    if nxt:
        st.info(f"Next up: Day {nxt[0]} · {logic.TASKS[nxt[1]['task']]} ({nxt[1]['difficulty'].title()})")
        st.button(f"Open {logic.TASKS[nxt[1]['task']]}", on_click=go, args=(TASK_PAGE[nxt[1]["task"]],), key="plan_open")
    else:
        st.success("Plan complete. Re-take each task and compare like-for-like on the Progress page.")

    for day in plan["days"]:
        n_done = sum(1 for it in day["items"] if it["id"] in done)
        marker = " (today)" if day["day"] == today_n else ""
        with st.expander(f"Day {day['day']} · {day['title']}{marker} - {n_done}/{len(day['items'])} done",
                         expanded=day["day"] == today_n):
            for it in day["items"]:
                wkey = f"plan_{pid}_{plan['created_at']}_{it['id']}"
                st.checkbox(f"{logic.TASKS[it['task']]} · {it['difficulty'].title()} · {it['minutes']} min",
                            value=it["id"] in done, key=wkey, on_change=_toggle_item,
                            args=(pid, it["id"], wkey))
                st.caption(it["purpose"])
    st.caption("Tick exercises yourself after completing them. Completion is self-reported, not detected automatically.")
    with st.expander("Replace this plan"):
        if st.button("Delete plan and start over", key="plan_del"):
            db.delete_plan(pid)
            st.rerun()


def page_progress(pid, df):
    st.header("📈 Progress")
    if df.empty:
        st.info("No results yet. Complete a task and your history will appear here.")
        quick_links("pr_empty")
        with st.expander("Presenting a demo? Load clearly labelled sample data"):
            sample_controls(pid, False, "pr_empty")
        return
    if df["is_sample"].any():
        st.warning("This view includes **sample data**.")

    plan = db.load_plan(pid)
    c = st.columns(3)
    c[0].metric("Completed sessions", len(df))
    if plan:
        total = sum(len(d["items"]) for d in plan["days"])
        c[1].metric("Plan completion", f"{len(plan.get('done', []))}/{total}")
    else:
        c[1].metric("Plan completion", "No plan")
    c[2].metric("Tasks tried", df["task"].nunique())

    task = st.selectbox("Task", list(logic.TASKS), format_func=logic.TASKS.get)
    rows = logic.task_rows(df, task)
    if rows.empty:
        st.info(f"No {logic.TASKS[task]} results yet.")
    else:
        rows = rows.assign(Attempt=list(range(1, len(rows) + 1)))
        rows["Difficulty"] = rows["difficulty"].str.title()
        l, r = st.columns(2)
        fig = px.line(rows, x="Attempt", y="score", color="Difficulty", markers=True,
                      color_discrete_sequence=PALETTE, title="Score by attempt")
        fig.update_yaxes(range=[0, 105])
        l.plotly_chart(style_fig(fig), use_container_width=True)
        ycol, ytitle = ("median_ms", "Median reaction (ms), lower is faster") if task == "reaction" \
            else ("accuracy", "Accuracy (%)")
        fig2 = px.line(rows, x="Attempt", y=ycol, color="Difficulty", markers=True,
                       color_discrete_sequence=PALETTE, title=ytitle)
        r.plotly_chart(style_fig(fig2), use_container_width=True)

        cmp = logic.compare_latest(rows)
        if cmp:
            prev, last = cmp
            st.subheader("Previous vs latest (same difficulty)")
            m = st.columns(2)
            m[0].metric("Score", f"{last['score']:.0f}", f"{last['score'] - prev['score']:+.0f}")
            if task == "reaction" and pd.notna(last["median_ms"]) and pd.notna(prev["median_ms"]):
                m[1].metric("Median reaction (ms)", f"{last['median_ms']:.0f}",
                            f"{last['median_ms'] - prev['median_ms']:+.0f}", delta_color="inverse")
            elif task != "reaction":
                m[1].metric("Accuracy", f"{float(last['accuracy']):.0f}%",
                            f"{float(last['accuracy']) - float(prev['accuracy']):+.0f}")
        else:
            st.caption("Two attempts at the same difficulty are needed for a like-for-like comparison.")

    st.info("Higher scores on a task mean better performance **on that task**. They are not proof of overall "
            "cognitive improvement. Repeating a task causes practice effects (you get better at the task itself), "
            "and results vary with device, mood, sleep and distraction. Compare like-for-like only.")

    with st.expander("Data tools"):
        export = df[["created_at", "task", "difficulty", "score", "accuracy", "duration_s", "is_sample"]]
        st.download_button("Download my results (CSV)", export.to_csv(index=False).encode("utf-8"),
                           "neurorevive_results.csv", "text/csv")
        sample_controls(pid, bool(df["is_sample"].any()), "pr")
        confirm = st.checkbox("I understand this permanently deletes all data for this participant ID",
                              key="del_confirm")
        if st.button("Delete all my data", disabled=not confirm, key="del_btn"):
            db.delete_participant_data(pid)
            st.rerun()


def page_reminders(pid, df):
    st.header("⏰ Reminders")
    saved = db.get_pref(pid, "reminder_time") or "18:00"
    try:
        h, m = (int(x) for x in saved.split(":"))
    except ValueError:
        h, m = 18, 0
    chosen = st.time_input("Preferred daily practice time", value=dtime(h, m), step=900)
    if st.button("Save preferred time", key="rem_save"):
        if db.set_pref(pid, "reminder_time", chosen.strftime("%H:%M")):
            st.success("Saved.")
    hhmm = chosen.strftime("%H:%M")

    plan = db.load_plan(pid)
    nxt = logic.next_plan_item(plan) if plan else None
    if nxt:
        suggestion = f"Day {nxt[0]} of your plan: {logic.TASKS[nxt[1]['task']]} ({nxt[1]['difficulty'].title()})"
    else:
        t, _ = logic.pick_next_task(df) if not df.empty else ("memory", "")
        suggestion = f"{logic.TASKS[t]} (no active plan; generate one on the Training Plan page)"
    st.markdown(f'<div class="nr-card"><span class="nr-pill">Next recommended session</span>'
                f"<h3>Daily at {hhmm}</h3><p>{suggestion}</p></div>", unsafe_allow_html=True)

    days = plan["length"] if plan else 7
    st.download_button("Download calendar reminder (.ics)", logic.make_ics(hhmm, days),
                       "neurorevive-reminder.ics", "text/calendar")
    st.caption("The .ics file adds a repeating daily event to your own calendar app, using your local time zone. "
               "This web app does not send push notifications or emails.")


def page_about():
    st.header("ℹ️ About & how it works")
    st.markdown(DISCLAIMER)
    with st.expander("The problem we're exploring", expanded=True):
        st.write("Educators have raised concerns about students' attention, memory and concentration. "
                 "This prototype explores whether short, repeatable tasks and personalised practice can help "
                 "people notice their own task performance and build a consistent practice habit. "
                 "We make no claim that it fixes or measures those concerns.")
    with st.expander("How the tests and scores work"):
        st.markdown(
            "- **Memory Matrix:** memorise highlighted cells, then tick them. Accuracy = targets found ÷ targets. "
            "Score = (targets found − false alarms) ÷ targets × 100.\n"
            "- **Focus Finder:** tick every target among look-alike distractors. Same accuracy/score formulas. "
            "Completion time is recorded but **not** part of the score. Each grid has a seed, so it can be reproduced.\n"
            "- **Reaction Rush:** 5 rounds. Score maps median reaction time (1200 ms → 0, 300 ms → 100) minus 10 per "
            "false start. Timing runs through the server and includes network delay.")
    with st.expander("Where the AI is"):
        st.markdown(
            "1. **Machine learning:** a scikit-learn linear regression is trained on your own score history "
            "(attempts at your latest difficulty, minimum 4) to forecast your next score and trend. "
            "It is trained per user, not pre-trained.\n"
            "2. **Rule engine:** transparent rules choose the next difficulty (below).\n"
            "3. **Optional language model:** if an API key is configured, a model explains the numbers in plain "
            "words. It only receives aggregated statistics.")
    with st.expander("How recommendations are generated"):
        st.markdown(
            "Interpretable rules (their IDs are shown next to each recommendation):\n"
            "- **R1/R2:** fewer than two attempts → complete an initial assessment first.\n"
            "- **R3:** the last two attempts at a level averaged ≥ 85% accuracy → suggest one level up.\n"
            "- **R4:** the last two attempts averaged < 50% → suggest one level down.\n"
            "- **R5:** otherwise repeat the level.\n"
            "- **R6–R8 (reaction):** many false starts → slow down; high variability → more rounds under "
            "consistent conditions; otherwise keep a steady routine.")
    with st.expander("What data is stored"):
        st.write("An anonymous participant ID, task name, difficulty, score, accuracy, timings, your training plan, "
                 "and your preferred reminder time. No names, emails or other personal details are requested. "
                 "There is no password: anyone who knows an ID can view that ID's results, so don't use your real "
                 "name. You can export or delete your data on the Progress page.")
    with st.expander("Limitations"):
        st.markdown(
            "- A handful of game scores is a noisy signal affected by practice, mood, sleep, device and distraction.\n"
            "- Getting better at a game is not evidence of overall cognitive improvement or decline.\n"
            "- Reaction timing in a browser has latency; it is not laboratory-grade.\n"
            "- Free hosting may reset stored data. Export your CSV if it matters.\n"
            "- This is not a medical device and not a diagnosis.")


# ------------------------------------------------------------------ main
def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
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

    df = db.load_results(pid)
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


main()
