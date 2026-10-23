"""Memory Matrix, Focus Finder and Reaction Rush pages."""
from __future__ import annotations

import random
import time

import pandas as pd
import streamlit as st

import db
import logic


# ------------------------------------------------------------------ shared helpers
def grid_html(size, correct, selected=None):
    """Highlighted grid (selected=None) or a result grid. Symbols as well as colour carry meaning."""
    cells = []
    for i in range(size * size):
        cls, mark = "", ""
        if selected is None:
            cls = "on" if i in correct else ""
        elif i in correct and i in selected:
            cls, mark = "hit", "✓"
        elif i in correct:
            cls, mark = "miss", "○"
        elif i in selected:
            cls, mark = "fp", "✕"
        cells.append(f'<div class="nr-cell {cls}">{mark}</div>')
    return (f'<div class="nr-grid" style="grid-template-columns:repeat({size},1fr)" '
            f'role="img" aria-label="{size} by {size} grid">' + "".join(cells) + "</div>")


def _result_metrics(res, duration):
    c = st.columns(4)
    c[0].metric("Score", f"{res['score']:.0f}")
    c[1].metric("Accuracy", f"{res['accuracy']:.0f}%")
    c[2].metric("Found", f"{res['hits']}/{res['targets']}")
    c[3].metric("False alarms", res["false_positives"])
    st.caption(f"Completion time: {duration:.1f} s (not included in the score).")


# ------------------------------------------------------------------ Memory Matrix
def _start_memory(s, diff):
    seed = random.randrange(1_000_000)
    s.clear()
    s.update(phase="show", diff=diff, seed=seed, nonce=str(time.time_ns()),
             cells=logic.generate_memory(diff, seed))


def memory_page(pid):
    st.header("🧩 Memory Matrix")
    s = st.session_state.setdefault("mem", {"phase": "start"})

    if s["phase"] == "start":
        st.write("A pattern of cells lights up briefly. When it disappears, tick the cells you remember.")
        diff = st.radio("Difficulty", logic.DIFFICULTIES, horizontal=True, key="mem_diff",
                        format_func=str.title)
        cfg = logic.MEMORY_CONFIG[diff]
        st.caption(f"{cfg['size']}×{cfg['size']} grid · {cfg['cells']} cells · shown for {cfg['show']} s")
        if st.button("Start round", type="primary", key="mem_start"):
            _start_memory(s, diff)
            st.rerun()

    elif s["phase"] == "show":
        cfg = logic.MEMORY_CONFIG[s["diff"]]
        st.info("Memorise the highlighted cells. They will disappear.")
        st.markdown(grid_html(cfg["size"], s["cells"]), unsafe_allow_html=True)
        time.sleep(cfg["show"])
        s["phase"] = "recall"
        s["recall_start"] = time.time()
        st.rerun()

    elif s["phase"] == "recall":
        size = logic.MEMORY_CONFIG[s["diff"]]["size"]
        key = lambda i: f"mem_{s['nonce']}_{i}"
        with st.form(f"mem_form_{s['nonce']}"):
            st.write("Tick the cells that were highlighted.")
            for r in range(size):
                cols = st.columns(size)
                for c in range(size):
                    cols[c].checkbox(f"Row {r + 1}, column {c + 1}", key=key(r * size + c),
                                     label_visibility="collapsed")
            submitted = st.form_submit_button("Submit answer", type="primary")
        if submitted:
            selected = {i for i in range(size * size) if st.session_state.get(key(i))}
            if not selected:
                st.warning("Select at least one cell before submitting.")
            else:
                duration = time.time() - s["recall_start"]
                res = logic.score_selection(s["cells"], selected)
                rid = db.save_result(pid, "memory", s["diff"], res["score"], res["accuracy"],
                                     round(duration, 1), {**res, "seed": s["seed"]})
                s.update(phase="result", result=res, selected=selected,
                         duration=duration, saved=rid is not None)
                st.rerun()

    elif s["phase"] == "result":
        size = logic.MEMORY_CONFIG[s["diff"]]["size"]
        _result_metrics(s["result"], s["duration"])
        if not s["saved"]:
            st.warning("This result could not be saved to the database.")
        st.markdown(grid_html(size, s["cells"], s["selected"]), unsafe_allow_html=True)
        st.caption("✓ correct · ○ missed · ✕ not highlighted (false alarm)")
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
    grid, targets = logic.generate_focus(diff, seed)
    s.clear()
    s.update(phase="play", diff=diff, seed=seed, nonce=str(time.time_ns()),
             grid=grid, targets=targets, started=time.time())


def focus_page(pid):
    st.header("🎯 Focus Finder")
    s = st.session_state.setdefault("foc", {"phase": "start"})

    if s["phase"] == "start":
        st.write("Find and tick every cell containing the target symbol, ignoring the look-alikes.")
        diff = st.radio("Difficulty", logic.DIFFICULTIES, horizontal=True, key="foc_diff",
                        format_func=str.title)
        cfg = logic.FOCUS_CONFIG[diff]
        st.caption(f"{cfg['size']}×{cfg['size']} grid · {cfg['targets']} targets · "
                   f"target '{cfg['target']}' among '{cfg['distractors']}'")
        seed_in = st.number_input("Seed (0 = random; the same seed gives the same grid)",
                                  min_value=0, max_value=999_999, value=0, step=1, key="foc_seed")
        if st.button("Start round", type="primary", key="foc_start"):
            _start_focus(s, diff, int(seed_in) or random.randrange(1, 1_000_000))
            st.rerun()

    elif s["phase"] == "play":
        cfg = logic.FOCUS_CONFIG[s["diff"]]
        size = cfg["size"]
        key = lambda i: f"foc_{s['nonce']}_{i}"
        st.info(f"Tick every cell containing **{cfg['target']}**. "
                f"Ignore look-alikes ({', '.join(cfg['distractors'])}). There are {cfg['targets']} targets.")
        with st.form(f"foc_form_{s['nonce']}"):
            for r in range(size):
                cols = st.columns(size)
                for c in range(size):
                    i = r * size + c
                    cols[c].checkbox(s["grid"][i], key=key(i), help=f"Row {r + 1}, column {c + 1}")
            submitted = st.form_submit_button("Submit answer", type="primary")
        if submitted:
            selected = {i for i in range(size * size) if st.session_state.get(key(i))}
            if not selected:
                st.warning("Select at least one cell before submitting.")
            else:
                duration = time.time() - s["started"]
                res = logic.score_selection(s["targets"], selected)
                rid = db.save_result(pid, "focus", s["diff"], res["score"], res["accuracy"],
                                     round(duration, 1), {**res, "seed": s["seed"]})
                s.update(phase="result", result=res, selected=selected,
                         duration=duration, saved=rid is not None)
                st.rerun()

    elif s["phase"] == "result":
        _result_metrics(s["result"], s["duration"])
        if not s["saved"]:
            st.warning("This result could not be saved to the database.")
        st.caption(f"Grid seed: {s['seed']}. Accuracy/score and completion time are reported separately. "
                   f"Missed targets: {s['result']['misses']}.")
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
    return {"phase": "idle", "round": 0, "rts": [], "false_starts": 0, "wait_until": 0.0,
            "last": "", "saved": False, "result": None, "save_ok": True, "last_seen": time.time()}


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
    s["phase"] = "done" if s["round"] >= logic.REACTION_ROUNDS else "idle"


def _render_panel():
    s = st.session_state["rr"]
    s["last_seen"] = time.time()
    phase = s["phase"]
    if phase == "idle":
        st.markdown(f"**Round {s['round'] + 1} of {logic.REACTION_ROUNDS}**")
        if s["last"]:
            st.info(s["last"])
        if st.button("Start round", type="primary", key="rr_start"):
            s["wait_until"] = time.time() + random.uniform(1.5, 4.0)
            s["phase"] = "waiting"
            st.rerun()
    elif phase == "waiting":
        st.markdown('<div class="rr-box rr-wait" role="status">Wait… do not click yet</div>',
                    unsafe_allow_html=True)
        if st.button("Click when the box turns green", key="rr_wait_btn", use_container_width=True):
            _register_click(s)
            st.rerun()
        elif time.time() >= s["wait_until"]:
            s["phase"] = "cue"
            st.rerun()
    elif phase == "cue":
        st.markdown('<div class="rr-box rr-go" role="status">GO! Click now</div>', unsafe_allow_html=True)
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

    st.write(f"{logic.REACTION_ROUNDS} rounds. Wait for the box to turn green, then click as fast as you can. "
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
        res = logic.score_reaction(s["rts"], s["false_starts"])
        rid = db.save_result(pid, "reaction", "standard", res["score"], res["accuracy"], None,
                             {"rts": s["rts"], "false_starts": s["false_starts"],
                              "rounds": logic.REACTION_ROUNDS, "median_ms": res["median_ms"]})
        s.update(saved=True, result=res, save_ok=rid is not None)
    res = s["result"]
    if not s["save_ok"]:
        st.warning("This result could not be saved to the database.")
    if not s["rts"]:
        st.warning("No valid reactions were recorded this session (all rounds were false starts).")
    else:
        c = st.columns(4)
        c[0].metric("Median", f"{res['median_ms']} ms")
        c[1].metric("Best", f"{res['best_ms']} ms")
        c[2].metric("False starts", s["false_starts"])
        c[3].metric("Score", f"{res['score']:.0f}")
        st.bar_chart(pd.DataFrame({"Reaction (ms)": s["rts"]}))
    if st.button("Run again", type="primary", key="rr_again"):
        st.session_state["rr"] = _new_rr()
        st.rerun()
