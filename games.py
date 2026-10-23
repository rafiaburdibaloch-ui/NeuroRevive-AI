"""Memory Matrix (digit sequence), Focus Finder and Reaction Rush pages. Uses core.py and style.css."""
from __future__ import annotations

import random
import statistics
import time
import uuid

import pandas as pd
import streamlit as st

import core

REACTION_ROUNDS = 5


# ------------------------------------------------------------------ Memory Matrix
def _start_memory(s, diff):
    seed = random.randrange(1_000_000)
    s.clear()
    s.update(phase="show", diff=diff, seed=seed, attempt=uuid.uuid4().hex,
             target=core.generate_sequence(diff, seed))


def memory_page(pid):
    st.header("🧩 Memory Matrix")
    s = st.session_state.setdefault("mem", {"phase": "start"})

    if s["phase"] == "start":
        st.write("A sequence of digits appears one at a time. When it ends, type the digits back in order.")
        diff = st.radio("Difficulty", core.LEVELS, horizontal=True, key="mem_diff", format_func=str.title)
        digits, secs = core.MEMORY_CFG[diff]
        st.caption(f"{digits} digits · {secs} s per digit")
        if st.button("Start round", type="primary", key="mem_start"):
            _start_memory(s, diff)
            st.rerun()

    elif s["phase"] == "show":
        secs = core.MEMORY_CFG[s["diff"]][1]
        st.info("Memorise the digits in order.")
        box = st.empty()
        for d in s["target"]:
            box.markdown(f'<div class="stimulus" role="status">{d}</div>', unsafe_allow_html=True)
            time.sleep(secs)
            box.markdown('<div class="stimulus">&nbsp;</div>', unsafe_allow_html=True)
            time.sleep(0.2)
        box.empty()
        s["phase"] = "recall"
        s["recall_start"] = time.time()
        st.rerun()

    elif s["phase"] == "recall":
        with st.form(f"mem_form_{s['attempt']}"):
            answer = st.text_input(f"Type the {len(s['target'])} digits in order", max_chars=len(s["target"]) + 4)
            submitted = st.form_submit_button("Submit answer", type="primary")
        if submitted:
            if not any(ch.isdigit() for ch in answer):
                st.warning("Enter at least one digit before submitting.")
            else:
                duration = time.time() - s["recall_start"]
                res = core.score_memory(s["target"], answer)
                ok = core.save_result(pid, "memory", s["diff"], res["score"],
                                      accuracy=res["correct"] / res["total"], duration_s=round(duration, 1),
                                      attempt=s["attempt"])
                s.update(phase="result", result=res, answer=answer, duration=duration, saved=ok)
                st.rerun()

    elif s["phase"] == "result":
        res = s["result"]
        c = st.columns(3)
        c[0].metric("Score", f"{res['score']:.0f}")
        c[1].metric("Digits correct", f"{res['correct']}/{res['total']}")
        c[2].metric("Recall time", f"{s['duration']:.1f} s")
        if not s["saved"]:
            st.warning("This result could not be saved to the database.")
        st.markdown(f"**Sequence:** `{s['target']}`  \n**Your answer:** `{s['answer'] or '—'}`")
        c1, c2 = st.columns(2)
        if c1.button("Play again", type="primary", key="mem_again"):
            _start_memory(s, s["diff"])
            st.rerun()
        if c2.button("Change difficulty", key="mem_change"):
            s.clear()
            s["phase"] = "start"
            st.rerun()


# ------------------------------------------------------------------ Focus Finder
def _start_focus(s, diff, seed):
    cells, targets = core.generate_grid(diff, seed)
    s.clear()
    s.update(phase="play", diff=diff, seed=seed, attempt=uuid.uuid4().hex,
             cells=cells, targets=targets, started=time.time())


def focus_page(pid):
    st.header("🎯 Focus Finder")
    s = st.session_state.setdefault("foc", {"phase": "start"})

    if s["phase"] == "start":
        st.write("Find and tick every cell containing the target letter, ignoring the look-alikes.")
        diff = st.radio("Difficulty", core.LEVELS, horizontal=True, key="foc_diff", format_func=str.title)
        rows, cols, k, target, distractors = core.FOCUS_CFG[diff]
        st.caption(f"{rows}×{cols} grid · {k} targets · target '{target}' among '{' '.join(distractors)}'")
        seed_in = st.number_input("Seed (0 = random; the same seed gives the same grid)",
                                  min_value=0, max_value=999_999, value=0, step=1, key="foc_seed")
        if st.button("Start round", type="primary", key="foc_start"):
            _start_focus(s, diff, int(seed_in) or random.randrange(1, 1_000_000))
            st.rerun()

    elif s["phase"] == "play":
        rows, cols, k, target, distractors = core.FOCUS_CFG[s["diff"]]
        key = lambda i: f"foc_{s['attempt']}_{i}"
        st.info(f"Tick every cell containing **{target}**. Ignore look-alikes ({', '.join(distractors)}). "
                f"There are {k} targets.")
        with st.form(f"foc_form_{s['attempt']}"):
            for r in range(rows):
                row_cols = st.columns(cols)
                for c in range(cols):
                    i = r * cols + c
                    row_cols[c].checkbox(s["cells"][i], key=key(i), help=f"Row {r + 1}, column {c + 1}")
            submitted = st.form_submit_button("Submit answer", type="primary")
        if submitted:
            picked = {i for i in range(rows * cols) if st.session_state.get(key(i))}
            if not picked:
                st.warning("Select at least one cell before submitting.")
            else:
                duration = time.time() - s["started"]
                res = core.score_focus(s["targets"], picked)
                ok = core.save_result(pid, "focus", s["diff"], res["score"], accuracy=res["accuracy"],
                                      duration_s=round(duration, 1), attempt=s["attempt"])
                s.update(phase="result", result=res, duration=duration, saved=ok)
                st.rerun()

    elif s["phase"] == "result":
        res = s["result"]
        k = len(s["targets"])
        c = st.columns(4)
        c[0].metric("Score (F1)", f"{res['score']:.0f}")
        c[1].metric("Accuracy", f"{res['accuracy'] * 100:.0f}%")
        c[2].metric("Found", f"{res['tp']}/{k}")
        c[3].metric("False alarms", res["fp"])
        st.caption(f"Completion time: {s['duration']:.1f} s (not included in the score). "
                   f"Missed targets: {res['fn']}. Grid seed: {s['seed']}.")
        if not s["saved"]:
            st.warning("This result could not be saved to the database.")
        c1, c2, c3 = st.columns(3)
        if c1.button("New grid", type="primary", key="foc_again"):
            _start_focus(s, s["diff"], random.randrange(1, 1_000_000))
            st.rerun()
        if c2.button("Retry this exact grid", key="foc_retry"):
            _start_focus(s, s["diff"], s["seed"])
            st.rerun()
        if c3.button("Change difficulty", key="foc_change"):
            s.clear()
            s["phase"] = "start"
            st.rerun()


# ------------------------------------------------------------------ Reaction Rush
def _new_rr():
    return {"phase": "idle", "round": 0, "rts": [], "false_starts": 0, "wait_until": 0.0, "last": "",
            "saved": False, "result": None, "save_ok": True, "last_seen": time.time(),
            "attempt": uuid.uuid4().hex}


def _register_click(s):
    now = time.time()
    if now < s["wait_until"]:
        s["false_starts"] += 1
        s["last"] = "False start: you clicked before the cue. Penalty applied."
    else:
        rt = (now - s["wait_until"]) * 1000
        if rt < 100:
            s["false_starts"] += 1
            s["last"] = "Too fast to be a reaction to the cue (anticipation). Counted as a false start."
        else:
            s["rts"].append(round(rt))
            s["last"] = f"Reaction: {round(rt)} ms"
    s["round"] += 1
    s["phase"] = "done" if s["round"] >= REACTION_ROUNDS else "idle"


def _render_panel():
    s = st.session_state["rr"]
    s["last_seen"] = time.time()
    phase = s["phase"]
    if phase == "idle":
        st.markdown(f'<div class="rx-box rx-idle" role="status">Round {s["round"] + 1} of '
                    f'{REACTION_ROUNDS}: get ready</div>', unsafe_allow_html=True)
        if s["last"]:
            st.info(s["last"])
        if st.button("Start round", type="primary", key="rr_start"):
            s["wait_until"] = time.time() + random.uniform(1.5, 4.0)
            s["phase"] = "waiting"
            st.rerun()
    elif phase == "waiting":
        st.markdown('<div class="rx-box rx-wait" role="status">Wait… do not click yet</div>',
                    unsafe_allow_html=True)
        if st.button("Click when the box turns green", key="rr_wait_btn", use_container_width=True):
            _register_click(s)
            st.rerun()
        elif time.time() >= s["wait_until"]:
            s["phase"] = "cue"
            st.rerun()
    elif phase == "cue":
        st.markdown('<div class="rx-box rx-go" role="status">GO! Click now</div>', unsafe_allow_html=True)
        if st.button("CLICK NOW", type="primary", key="rr_cue_btn", use_container_width=True):
            _register_click(s)
            st.rerun()


@st.fragment(run_every=0.2)
def _polling_panel():
    _render_panel()


@st.fragment
def _static_panel():
    _render_panel()


def reaction_page(pid):
    st.header("⚡ Reaction Rush")
    s = st.session_state.setdefault("rr", _new_rr())
    if s["phase"] in ("waiting", "cue") and time.time() - s["last_seen"] > 3:
        s["phase"] = "idle"
        s["last"] = "The previous round was interrupted and was not counted."
    s["last_seen"] = time.time()

    st.write(f"{REACTION_ROUNDS} rounds. Wait for the box to turn green, then click as fast as you can. "
             "Clicking early counts as a false start and lowers your score.")
    st.caption("Measurement note: timing runs through the server, so network and browser delay "
               "(often 100-300 ms) is included. Compare results only with your own results on the "
               "same device. This is not a laboratory measurement.")

    if s["phase"] == "done":
        _show_summary(pid, s)
    elif s["phase"] == "waiting":
        _polling_panel()
    else:
        _static_panel()


def _show_summary(pid, s):
    if not s["saved"]:
        res = core.score_reaction(s["rts"], s["false_starts"])
        ok = core.save_result(pid, "reaction", "standard", res["score"], accuracy=res["accuracy"],
                              rt_ms=res["mean"], rt_sd=res["sd"], attempt=s["attempt"])
        s.update(saved=True, result=res, save_ok=ok)
    res = s["result"]
    if not s["save_ok"]:
        st.warning("This result could not be saved to the database.")
    if not s["rts"]:
        st.warning("No valid reactions were recorded this session (all rounds were false starts).")
    else:
        c = st.columns(4)
        c[0].metric("Median", f"{statistics.median(s['rts']):.0f} ms")
        c[1].metric("Best", f"{min(s['rts'])} ms")
        c[2].metric("False starts", s["false_starts"])
        c[3].metric("Score", f"{res['score']:.0f}")
        st.bar_chart(pd.DataFrame({"Reaction (ms)": s["rts"]}))
    if st.button("Run again", type="primary", key="rr_again"):
        st.session_state["rr"] = _new_rr()
        st.rerun()
