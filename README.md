# 🧠 NeuroRevive AI

**An AI-assisted cognitive task practice and self-monitoring prototype.**

**Live demo:** https://neurorevive.streamlit.app/

> **Important:** NeuroRevive AI is a prototype for cognitive task practice and self-monitoring. It does **not** diagnose any condition, measure intelligence or brain health, or provide clinically validated rehabilitation. If you have health concerns, please speak to a qualified professional.

---

## The problem

Educators have raised concerns about students' attention, memory and concentration. NeuroRevive AI explores whether short, repeatable cognitive tasks and personalised practice can help people notice how they perform on specific tasks and build a consistent practice habit. We make no claim that it fixes or measures those concerns.

## Features

| Area | What it does |
|---|---|
| **Dashboard** | Sessions completed, latest and average scores, practice streak, trend chart, recommended next activity, drop-off check, and quick links to each task. Helpful empty state for new users. |
| **Memory Matrix** | Digits appear one at a time, then you type them back. Easy, medium and hard levels. Score = % of digits in the correct position. |
| **Focus Finder** | Tick every target letter among look-alike distractors. Grids are generated from a seed, so a round can be reproduced. Score = F1 of ticked cells (misses and false alarms both count). Completion time is recorded but not scored. |
| **Reaction Rush** | 5 rounds with a random wait before the cue. Early clicks are false starts and are penalised. Reports median, best, false starts and score. |
| **Analysis** | Rule-based recommendations, a per-task machine-learning forecast, and an optional AI coach. |
| **Training Plan** | 7-day or 14-day plans (rule-based or AI-written). Tick exercises off; progress is saved and survives a refresh. Last day is a reassessment. |
| **Progress** | Score and accuracy / reaction-time trends per task, previous-vs-latest comparison at the same difficulty, plan completion, CSV export and data deletion. |
| **Reminders** | Pick a daily practice time and download a repeating `.ics` calendar event. (No push notifications are sent.) |
| **Accounts** | Guest sessions with private random IDs, plus username + password accounts so people only see their own data. |
| **About** | Explains scoring, recommendations, stored data and limitations. |

## Where the AI is

1. **Rule engine (transparent baseline):** interpretable rules choose the next difficulty and activity from recent like-for-like results (for example, a high average moves up one level, a low average moves down one, too few attempts asks for an initial assessment).
2. **Machine learning:** a scikit-learn `LinearRegression` is trained per user, per task, on their own last attempts at their current difficulty (minimum 4) to forecast the next score and trend. It is trained on the user's own data, not pre-trained.
3. **Language models (optional):** if an API key is configured, a model writes short coaching feedback and can draft a 7/14-day plan, in English or Urdu. The model only receives **aggregated statistics**; no username or raw rows are sent. Plan output is validated (task names, difficulty, minutes, banned medical wording) before it is saved.

The app works fully without any API key. In that case only rules and the ML forecast are used, and the UI says so.

## Architecture

```
NeuroRevive-AI/
├── app.py            # Streamlit entry point: pages, navigation, accounts, charts
├── games.py          # Memory Matrix, Focus Finder and Reaction Rush pages
├── core.py           # SQLite, scoring, rule engine, ML forecast, LLM layer (no Streamlit code)
├── style.css         # Custom dark theme (midnight blue, indigo, violet, cyan)
├── requirements.txt
└── README.md
```

- **UI:** Streamlit with custom CSS (`style.css`) and Plotly charts.
- **Storage:** SQLite (`neurorevive.db`) with tables for results, plans, completion ticks, reminder time and users. Each result has a unique attempt ID, so one submission can never be saved twice.
- **LLM layer:** Gemini first, then Groq as a fallback, then rule-based results. Model names can be overridden with secrets.

## Setup and run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to a GitHub repository.
2. Go to https://share.streamlit.io, choose **Create app**, select the repo, branch `main` and main file `app.py`.
3. (Optional) Add AI keys under **Settings → Secrets**:


## How scores work

- **Memory Matrix:** (digits in the correct position ÷ total digits) × 100.
- **Focus Finder:** F1 score of your ticked cells vs. the true targets. Accuracy = hits ÷ (hits + false alarms + misses).
- **Reaction Rush:** 100 − (mean reaction ms − 250) ÷ 6 − 8 per false start, limited to 0–100. Reactions under 100 ms are treated as anticipation.

## What data is stored

A username (or random guest ID), a salted and hashed password if you create an account, task, difficulty, score, accuracy, timings, your plans and tick marks, and your reminder time. Users can export their results as CSV and delete their data on the Progress page.

## Limitations

- A handful of game scores is a **noisy** signal affected by practice, mood, sleep, device and distraction.
- Getting better at a game is **not** evidence of overall cognitive improvement or decline.
- Reaction timing runs through the server and includes network delay; it is not laboratory-grade.
- The login is a lightweight prototype system, not production-grade security. Do not use your real name or a password you use elsewhere.
- SQLite on free hosting can reset when the app restarts, which also removes accounts. A production version would use a hosted database (for example Supabase or Postgres).
- Not clinically validated and not a medical device.

## Future work

- Hosted database and production-grade authentication.
- A small study with real students to test whether practice habits improve.
- Teacher view with student-controlled sharing.
- Automated tests for scoring and recommendation rules.

## Team

Surhan Mehak Memon
Muzaena Shahid
Rafia Burdi Baloch - Leader


