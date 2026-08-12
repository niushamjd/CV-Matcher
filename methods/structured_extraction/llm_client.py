"""
LLM client wrapper -- provider-agnostic so the team can swap backends easily.

DEFAULT: Groq (100% free tier, no cost, much faster and stronger than local
CPU inference since it runs on Groq's own hardware).
    Setup: sign up for a free API key at https://console.groq.com
    Then set it as an environment variable:
        setx GROQ_API_KEY "your_key_here"     (Windows, persists across sessions)
    No credit card required for the free tier.

A local Ollama fallback is also included below (100% free, fully offline,
no API key) in case you ever want to run without internet -- just flip
LLM_PROVIDER to "ollama".
"""

import json
import os
import time
import requests

# --- Config: change this one line to switch providers ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq", "gemini", or "ollama"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = "http://localhost:11434/api/chat"


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
    """Returns the raw text response from the LLM. Callers parse it further."""
    if LLM_PROVIDER == "groq":
        return _call_groq(system_prompt, user_prompt, json_mode)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_prompt, json_mode)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, user_prompt, json_mode)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _call_groq(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    """
    Groq's free tier: sign up at https://console.groq.com, no cost, no card needed.
    Runs on Groq's hardware (LPUs) -- much faster than local CPU inference,
    and the 70B model is meaningfully stronger than what most laptops can run locally.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com, "
            "then run: setx GROQ_API_KEY \"your_key_here\" (Windows) and restart your terminal."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,  # low temperature: we want consistent structured output, not creativity
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    if resp.status_code == 429:
        # Rate limited — retry with longer waits to clear the 60s token window
        for attempt, wait in enumerate([30, 60, 90], start=1):
            print(f"  [rate limit] waiting {wait}s before retry {attempt}/3...")
            time.sleep(wait)
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code != 429:
                break
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    """
    Gemini via OpenAI-compatible SDK — identical setup to Niyousha's llm_judge_starter.py.
    1M tokens/minute on the free tier, no practical rate limit for 30 pairs.
    """
    from openai import OpenAI

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com, "
            "then run: export GEMINI_API_KEY=\"your_key_here\""
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        temperature=0.2,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# --- Fallback: fully local/offline option (no API key, needs Ollama installed) ---
def _call_ollama(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Is Ollama installed and running? Try 'ollama pull llama3.1:8b' first."
        )

    data = response.json()
    return data["message"]["content"]


_WRITING_QUALITY_PROMPT = """Rate the writing quality of this CV on clarity,
grammar, and professionalism only — NOT on the candidate's actual skills or
experience level. A junior candidate with clean, clear writing should score
as well as a senior candidate with clean writing.

Return ONLY valid JSON, no other text:
{
  "writing_quality_score": <int 0-100, where 100 = very clear/professional writing>,
  "explanation": "<1 sentence reason>"
}
"""


def writing_quality_signal(cv_text: str) -> dict:
    """
    Returns {"writing_quality_score": 0-100, "explanation": str}.
    Mirrors Niyousha's writing_quality_signal() — same prompt, provider-agnostic.
    """
    import re
    raw = call_llm(
        system_prompt=_WRITING_QUALITY_PROMPT,
        user_prompt=cv_text,
        json_mode=True,
    ).strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    parsed = json.loads(raw)
    return {
        "writing_quality_score": parsed.get("writing_quality_score"),
        "explanation": parsed.get("explanation", ""),
    }


if __name__ == "__main__":
    out = call_llm(
        system_prompt="Reply with valid JSON only: {\"status\": \"ok\"}",
        user_prompt="ping",
        json_mode=True,
    )
    print("Raw response:", out)