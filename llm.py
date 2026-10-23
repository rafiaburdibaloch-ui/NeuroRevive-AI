"""Optional LLM explanation. Only aggregated numbers are sent, never identifiers."""
import requests

DEFAULT_MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM = (
    "You write short, encouraging practice feedback for a cognitive-task practice prototype. "
    "You are given aggregated game statistics and rule-based recommendations. "
    "Explain them in plain language in under 180 words and give 2-3 concrete practice tips. "
    "Rules: never diagnose, never comment on intelligence, brain health, ADHD, dementia or any medical "
    "or psychological condition, never claim the scores prove improvement or decline, mention that "
    "a few game scores are noisy and affected by practice, device and mood, and stay consistent with "
    "the recommended difficulty levels provided."
)


def explain(payload_text: str, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 25) -> str:
    resp = requests.post(
        API_URL,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 500, "system": SYSTEM,
              "messages": [{"role": "user", "content": payload_text}]},
        timeout=timeout,
    )
    resp.raise_for_status()
    blocks = resp.json().get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    if not text:
        raise ValueError("Empty response from model")
    return text
