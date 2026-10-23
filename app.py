"""NeuroRevive AI - Streamlit app. Prototype for cognitive task practice, NOT a diagnostic tool."""
import os
import random
import re
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import core

st.set_page_config(page_title="NeuroRevive AI", page_icon="🧠", layout="wide")
core.init_db()
st.markdown(f"<style>{Path(__file__).with_name('styles.css').read_text()}</style>",
            unsafe_allow_html=True)

PAGES = ["Dashboard", "Memory Matrix", "Focus Finder", "Reaction Rush", "Analysis",
         "Training Plan", "Progress", "Reminders", "About"]
PID_RE = re.compile(r"[A-Za-z0-9_-]{3,30}")
COLORS = {"Memory Matrix": "#a78bfa", "Focus Finder": "#22d3ee", "Reaction Rush": "#818cf8"}
DISCLAIMER = ("⚠️ Prototype for cognitive task practice and self-monitoring. It does not diagnose "
              "any condition, measure intelligence, or provide clinically validated rehabilitation.")

S = st.session_state
S.setdefault("page", "Dashboard")
S.setdefault("pid", "guest-" + uuid.uuid4().hex[:6])


def go(page):
    S.page = page


def fmt(v, suffix=""):
    return "—" if v is None or pd.isna(v) else f"{v:.0f}{suffix}"


def hero(title, sub):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{sub}</p></div>", unsafe_allow_html=True)


def card(title, body, badge=None):
    b = f"<span class='badge violet'>{badge}</span>" if badge else ""
    st.markdown(f"<div class='card'><h4>{b}{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)


def disclaimer():
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


def line_chart(df, y, title, ytitle):
    d = df.dropna(subset=[y])
    if d.empty:
        return
    fig = px.line(d, x="created", y=y, color="task_name", markers=True, title=title,
                  color_discrete_map=COLORS, template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="", yaxis_title=ytitle, xaxis_title=None, height=340)
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🧠 NeuroRevive AI")
st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
pid_in = st.sidebar.text_input("Participant ID (nickname, no real names)", value=S.pid, max_chars=30)
if PID_RE.fullmatch(pid_in):
    S.pid = pid_in
else:
    st.sidebar.error("Use 3-30 letters, numbers, - or _")
pid = S.pid
st.sidebar.caption("Demo participant mode: not a login. Anyone who knows an ID can open it.")
with st.sidebar.expander("Demo tools"):
    if st.button("Load labelled sample data"):
        core.generate_sample(pid)
        st.rerun()
    if st.button("Remove sample data"):
        core.delete_sample(pid)
        st.rerun()


# ------------------------------------------------------------------ dashboard
def page_dashboard():
    df = core.get_results(pid)
    hero("NeuroRevive AI", "Interactive cognitive mini-tasks, personalised practice routines and "
         "honest progress tracking.")
    if df.empty:
        card("Welcome! No results yet",
             "Take your first mini-task below. Nothing is shown until you have real (or clearly "
             "labelled sample) results.", "Getting started")
    else:
        if df["sample"].any():
            st.warning("This profile contains labelled SAMPLE data (demo tools in the sidebar).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions completed", len(df))
        c2.metric("Latest score", fmt(df["score"].iloc[-1]))
        c3.metric("Average score", fmt(df["score"].mean()))
        c4.metric("Practice streak (days)", core.current_streak(df))
        left, right = st.columns([3, 2])
        with left:
            line_chart(df, "score", "Recent task scores", "Score (0-100)")
        with right:
            st.markdown("**Recent activity**")
            recent = df.tail(6).iloc[::-1][["created", "task_name", "difficulty", "score"]]
            recent["created"] = recent["created"].dt.strftime("%d %b %H:%M")
            recent.columns = ["When (UTC)", "Task", "Level", "Score"]
            st.dataframe(recent, hide_index=True, use_container_width=True)
    a = core.analyse(df)
    nxt = core.next_activity(df, a["recs"]) if not df.empty else a["recs"][0]
    card(f"Recommended next: {core.TASKS[nxt['task']]} ({nxt['difficulty']})", nxt["reason"],
         a["source"])
    st.markdown("### Start a task")
    cols = st.columns(3)
    for col, (name, desc) in zip(cols, [
            ("Memory Matrix", "Memorise a digit sequence, then recall it."),
            ("Focus Finder", "Spot every target among look-alike distractors."),
            ("Reaction Rush", "React to the green cue; avoid false starts.")]):
        with col:
            card(name, desc)
            st.button(f"Open {name}", key=f"go_{name}", on_click=go, args=(name,),
                      use_container_width=True)
    disclaimer()


# --------------------------------------------------------------- memory matrix
def page_memory():
    hero("Memory Matrix", "Memorise the digits. They disappear. Type them back.")
    m = S.setdefault("mem", {"stage": "idle"})
    if m["stage"] == "idle":
        level = st.radio("Difficulty", core.LEVELS, horizontal=True, format_func=str.title)
        n, sec = core.MEMORY_CFG[level]
        st.info(f"You'll see {n} digits for {n * sec:.1f} seconds, then type them from memory.")
        if st.button("Start", type="primary"):
            seed = random.randint(1, 10**9)
            m.update(stage="show", level=level, seed=seed, seq=core.generate_sequence(level, seed),
                     attempt=uuid.uuid4().hex)
            st.rerun()
    elif m["stage"] == "show":
        n, sec = core.MEMORY_CFG[m["level"]]
        ph = st.empty()
        ph.markdown(f"<div class='stimulus'>{' '.join(m['seq'])}</div>", unsafe_allow_html=True)
        time.sleep(n * sec)
        ph.empty()
        m.update(stage="answer", t0=time.time())
        st.rerun()
    elif m["stage"] == "answer":
        st.markdown("<div class='stimulus'>? ? ?</div>", unsafe_allow_html=True)
        with st.form("mem_form"):
            ans = st.text_input("Type the digits you saw", max_chars=20)
            ok = st.form_submit_button("Submit", type="primary")
        if ok:
            if not any(ch.isdigit() for ch in ans):
                st.warning("Please enter at least one digit.")
            else:
                dur = time.time() - m["t0"]
                res = core.score_memory(m["seq"], ans)
                core.save_result(pid, "memory", m["level"], res["score"],
                                 accuracy=res["score"] / 100, duration_s=round(dur, 1),
                                 attempt=m["attempt"])
                m.update(stage="result", res=res, ans=ans, dur=dur)
                st.rerun()
    else:
        res = m["res"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{res['score']:.0f}")
        c2.metric("Digits correct", f"{res['correct']}/{res['total']}")
        c3.metric("Recall time", f"{m['dur']:.1f}s")
        st.write(f"Sequence: `{m['seq']}`  ·  Your answer: `{m['ans']}`")
        st.success("Result saved.")
        if st.button("Play again", type="primary"):
            S["mem"] = {"stage": "idle"}
            st.rerun()
    disclaimer()


# ---------------------------------------------------------------- focus finder
def page_focus():
    hero("Focus Finder", "Find every target among the distractors.")
    f = S.setdefault("foc", {"stage": "idle"})
    if f["stage"] == "idle":
        level = st.radio("Difficulty", core.LEVELS, horizontal=True, format_func=str.title)
        rows, cols, k, target, _ = core.FOCUS_CFG[level]
        st.info(f"Tick every **{target}** in a {rows}×{cols} grid ({k} targets hidden). "
                f"Accuracy decides your score; time is shown separately.")
        if st.button("Start", type="primary"):
            seed = random.randint(1, 10**9)
            cells, targets = core.generate_grid(level, seed)
            f.update(stage="play", level=level, seed=seed, cells=cells, targets=targets,
                     attempt=uuid.uuid4().hex, t0=time.time())
            st.rerun()
    elif f["stage"] == "play":
        rows, cols, k, target, _ = core.FOCUS_CFG[f["level"]]
        st.markdown(f"### Target: <span class='badge'>{target}</span> — tick every {target}",
                    unsafe_allow_html=True)
        with st.form("foc_form"):
            picks = {}
            for r in range(rows):
                cs = st.columns(cols)
                for c in range(cols):
                    i = r * cols + c
                    picks[i] = cs[c].checkbox(f["cells"][i], key=f"foc_{f['attempt']}_{i}")
            ok = st.form_submit_button("Submit", type="primary")
        if ok:
            dur = time.time() - f["t0"]
            picked = [i for i, v in picks.items() if v]
            res = core.score_focus(f["targets"], picked)
            core.save_result(pid, "focus", f["level"], res["score"], accuracy=res["accuracy"],
                             duration_s=round(dur, 1), attempt=f["attempt"])
            f.update(stage="result", res=res, dur=dur)
            st.rerun()
    else:
        res = f["res"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score (accuracy)", f"{res['score']:.0f}")
        c2.metric("Targets found", res["tp"])
        c3.metric("Missed / wrong picks", f"{res['fn']} / {res['fp']}")
        c4.metric("Time (not scored)", f"{f['dur']:.1f}s")
        st.success("Result saved.")
        if st.button("Retry", type="primary"):
            S["foc"] = {"stage": "idle"}
            st.rerun()
    disclaimer()


# --------------------------------------------------------------- reaction rush
RX_ROUNDS = 5


def new_rx():
    return {"stage": "idle", "round": 1, "cue_at": 0.0, "rts": [], "false": 0, "note": "",
            "attempt": uuid.uuid4().hex, "t0": 0.0, "summary": None}


def _schedule(rx):
    rx["cue_at"] = time.time() + random.uniform(1.5, 4.0)


@st.fragment(run_every=0.1)
def reaction_fragment():
    rx = S["rx"]
    if rx["stage"] == "wait" and time.time() >= rx["cue_at"]:
        rx["stage"] = "cue"
    stage = rx["stage"]
    if stage == "idle":
        st.markdown("<div class='rx-box rx-idle'>Ready?</div>", unsafe_allow_html=True)
        if st.button("Start test", type="primary", key="rx_start"):
            rx.update(stage="wait", t0=time.time())
            _schedule(rx)
            st.rerun(scope="fragment")
    elif stage == "wait":
        st.markdown("<div class='rx-box rx-wait'>Wait for green…</div>", unsafe_allow_html=True)
        st.caption(f"Round {rx['round']} of {RX_ROUNDS}  ·  {rx['note']}")
        if st.button("Click", key=f"rxw_{rx['round']}_{rx['false']}"):
            rx["false"] += 1
            rx["note"] = "False start! Wait for green."
            _schedule(rx)
            st.rerun(scope="fragment")
    elif stage == "cue":
        st.markdown("<div class='rx-box rx-go'>CLICK NOW!</div>", unsafe_allow_html=True)
        st.caption(f"Round {rx['round']} of {RX_ROUNDS}")
        if st.button("Click", type="primary", key=f"rxc_{rx['round']}"):
            rt = (time.time() - rx["cue_at"]) * 1000
            if rt < 100:  # implausibly fast = anticipation, count as false start
                rx["false"] += 1
                rx["note"] = "Too fast to be a reaction: counted as a false start."
                rx["stage"] = "wait"
                _schedule(rx)
            else:
                rx["rts"].append(rt)
                rx["note"] = f"Round {rx['round']}: {rt:.0f} ms"
                rx["round"] += 1
                if rx["round"] > RX_ROUNDS:
                    res = core.score_reaction(rx["rts"], rx["false"])
                    core.save_result(pid, "reaction", "standard", res["score"],
                                     accuracy=res["accuracy"], rt_ms=res["mean"], rt_sd=res["sd"],
                                     duration_s=round(time.time() - rx["t0"], 1),
                                     attempt=rx["attempt"])
                    rx.update(stage="done", summary=res)
                else:
                    rx["stage"] = "wait"
                    _schedule(rx)
            st.rerun(scope="fragment")
    else:
        res = rx["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{res['score']:.0f}")
        c2.metric("Mean reaction", fmt(res["mean"], " ms"))
        c3.metric("Variation (SD)", fmt(res["sd"], " ms"))
        c4.metric("False starts", rx["false"])
        st.write("Rounds: " + ", ".join(f"{r:.0f} ms" for r in rx["rts"]))
        st.success("Result saved.")
        if st.button("Run again", type="primary", key="rx_again"):
            S["rx"] = new_rx()
            st.rerun(scope="fragment")


def page_reaction():
    hero("Reaction Rush", f"{RX_ROUNDS} rounds. Click only when the box turns green.")
    S.setdefault("rx", new_rx())
    st.info("Times are measured in the app server and include network and browser delay, so they "
            "are approximate (not laboratory-grade) and only comparable on the same device and "
            "connection.")
    reaction_fragment()
    disclaimer()


# -------------------------------------------------------------------- analysis
def get_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY")


def page_analysis():
    hero("Performance analysis", "Built from your saved task results, using transparent rules.")
    df = core.get_results(pid)
    if df.empty:
        st.info("No results yet. Complete a task first.")
        return
    a = core.analyse(df)
    st.markdown(f"<span class='badge violet'>{a['source']}</span>", unsafe_allow_html=True)
    st.markdown("#### What your saved results show")
    for line in a["findings"] or ["Not enough attempts yet for a summary."]:
        st.write("• " + line)
    st.markdown("#### Personalised practice recommendations")
    badge = {"up": "Step up", "down": "Ease off", "keep": "Keep going",
             "assess": "Initial assessment", "consistency": "Consistency"}
    for r in a["recs"]:
        card(f"{core.TASKS[r['task']]} → {r['difficulty']}", r["reason"], badge[r["action"]])
    st.caption(a["caveat"])
    st.markdown("#### Optional: AI-written explanation")
    key = get_key()
    if not key:
        st.info("No API key configured, so only the rule-based analysis above is used. "
                "(Add ANTHROPIC_API_KEY in app secrets to enable this.)")
    elif st.button("Generate AI explanation (sends aggregate scores only)"):
        summary = "\n".join(a["findings"] + [f"{r['task']}: {r['reason']}" for r in a["recs"]])
        try:
            with st.spinner("Asking the model..."):
                text = core.llm_explain(summary, key, os.getenv("NEUROREVIVE_MODEL", "claude-sonnet-5-5"))
            st.markdown("<span class='badge'>AI-generated</span>", unsafe_allow_html=True)
            st.write(text)
        except Exception as e:  # graceful fallback
            st.warning(f"AI call failed ({type(e).__name__}). Showing rule-based advice only.")
    disclaimer()


# ------------------------------------------------------------------------ plan
def _toggle(days, day, item, key):
    core.set_done(pid, days, day, item, bool(S[key]))


def page_plan():
    hero("Training plan", "A short daily routine built from your results.")
    df = core.get_results(pid)
    a = core.analyse(df)
    days = st.radio("Plan length", [7, 14], horizontal=True, format_func=lambda d: f"{d}-day")
    if st.button("Generate / regenerate plan (resets progress)", type="primary"):
        plan, label = core.build_plan(days, a["recs"])
        core.save_plan(pid, days, plan, label)
        st.rerun()
    saved = core.get_plan(pid, days)
    if not saved:
        st.info("No plan yet. Click the button above.")
        return
    plan, label = saved
    done = core.get_done(pid, days)
    total = sum(len(d["items"]) for d in plan)
    st.markdown(f"<span class='badge amber'>{label}</span>", unsafe_allow_html=True)
    if label == "Starter plan":
        st.caption("Starter plan: you need at least 2 attempts per task before it can be personalised.")
    st.progress(len(done) / total, text=f"{len(done)} of {total} exercises completed")
    for d in plan:
        n_done = sum(1 for i in range(len(d["items"])) if (d["day"], i) in done)
        with st.expander(f"Day {d['day']}: {n_done}/{len(d['items'])} done"):
            for i, it in enumerate(d["items"]):
                key = f"plan_{pid}_{days}_{d['day']}_{i}"
                st.checkbox(f"{core.TASKS[it['task']]} · {it['difficulty']} · {it['minutes']} min. "
                            f"{it['purpose']}", value=(d["day"], i) in done, key=key,
                            on_change=_toggle, args=(days, d["day"], i, key))
    disclaimer()


# -------------------------------------------------------------------- progress
def page_progress():
    hero("Progress", "Compare like with like. Higher scores are not proof of overall improvement.")
    df = core.get_results(pid)
    if df.empty:
        st.info("No results yet. Complete a task to see your progress.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions", len(df))
    for col, d in zip((c2, c3), (7, 14)):
        pc = core.plan_completion(pid, d)
        col.metric(f"{d}-day plan", f"{pc[0]}/{pc[1]}" if pc else "no plan")
    line_chart(df, "score", "Score over time", "Score (0-100)")
    line_chart(df, "accuracy", "Accuracy over time", "Accuracy (0-1)")
    line_chart(df, "rt_ms", "Mean reaction time (lower = faster)", "ms")
    st.markdown("#### Previous vs latest attempt")
    cols = st.columns(3)
    for col, (t, name) in zip(cols, core.TASKS.items()):
        d = df[df["task"] == t]
        with col:
            if len(d) < 2:
                st.metric(name, fmt(d["score"].iloc[-1]) if len(d) else "—", help="Need 2+ attempts")
            elif d["difficulty"].iloc[-1] != d["difficulty"].iloc[-2]:
                st.metric(name, fmt(d["score"].iloc[-1]))
                st.caption("Last two attempts were at different difficulty, so not directly comparable.")
            else:
                st.metric(name, fmt(d["score"].iloc[-1]),
                          f"{d['score'].iloc[-1] - d['score'].iloc[-2]:+.0f} vs previous")
    st.caption("Early gains often reflect practice effects (getting used to the task). Short games "
               "are noisy, so treat trends as practice feedback only.")
    st.download_button("Download my results (CSV backup)", df.to_csv(index=False), "neurorevive_results.csv")
    disclaimer()


# ------------------------------------------------------------------- reminders
def page_reminders():
    hero("Reminders", "Pick a daily practice time and add it to your own calendar.")
    df = core.get_results(pid)
    a = core.analyse(df)
    cur = core.get_time(pid)
    t = st.time_input("Preferred daily practice time",
                      value=pd.to_datetime(cur, format="%H:%M").time())
    if st.button("Save time", type="primary"):
        core.set_time(pid, t.strftime("%H:%M"))
        st.success("Saved.")
    nxt = core.next_activity(df, a["recs"]) if not df.empty else a["recs"][0]
    card("Next recommended session",
         f"Daily at {t.strftime('%H:%M')}: {core.TASKS[nxt['task']]} ({nxt['difficulty']}).")
    days = st.radio("Calendar reminder length", [7, 14], horizontal=True)
    st.download_button("Download calendar reminder (.ics)",
                       core.make_ics(t.strftime("%H:%M"), days, core.TASKS[nxt["task"]]),
                       "neurorevive_reminder.ics", "text/calendar")
    st.caption("This app cannot send push notifications. The .ics file adds repeating reminders to "
               "your own calendar app, which then notifies you.")


# ----------------------------------------------------------------------- about
def page_about():
    hero("About & responsible use", "How NeuroRevive AI works and what it can't do.")
    st.markdown("""
**The problem.** Educators worry that students struggle with attention, memory and concentration.
NeuroRevive AI explores whether short, repeatable tasks and personalised practice help people
understand their own task performance and build a practice habit.

**The tasks**
- **Memory Matrix:** score = % of digits recalled in the correct position.
- **Focus Finder:** score = F1 of ticked vs true targets (penalises misses and wrong picks). Time is shown separately.
- **Reaction Rush:** score = 100 − (mean ms − 250)/6 − 8 per false start. Five rounds, random 1.5–4 s waits.

**How recommendations work.** A rule-based engine looks at your last three attempts per task: average ≥ 85 → try the next difficulty; < 55 → step down; otherwise stay. Very variable reaction times → extra rounds in consistent conditions. Fewer than 2 attempts → initial assessment (starter plan). An optional language model may reword these findings; it only sees aggregate numbers.

**Data stored.** A nickname you choose, task scores, difficulty, timestamps, plan progress and preferred practice time. No real names, emails or health information.

**Limitations.** Reaction timing includes network delay. Short games are noisy, practice effects inflate early gains, and results depend on device, sleep, and mood. Prototype storage may reset when the demo app restarts.

**Not a diagnosis.** This is not a medical device. It cannot diagnose any condition or say anything about intelligence or brain health. If you are concerned about attention or memory, talk to a qualified professional.
""")


{"Dashboard": page_dashboard, "Memory Matrix": page_memory, "Focus Finder": page_focus,
 "Reaction Rush": page_reaction, "Analysis": page_analysis, "Training Plan": page_plan,
 "Progress": page_progress, "Reminders": page_reminders, "About": page_about}[S.page]()
