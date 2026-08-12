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
import requests

# --- Config: change this one line to switch providers ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" or "ollama"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = "http://localhost:11434/api/chat"


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
    """Returns the raw text response from the LLM. Callers parse it further."""
    if LLM_PROVIDER == "groq":
        return _call_groq(system_prompt, user_prompt, json_mode)
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
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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


if __name__ == "__main__":
    # Quick connectivity test
    out = call_llm(
        system_prompt="Reply with valid JSON only: {\"status\": \"ok\"}",
        user_prompt="ping",
        json_mode=True,
    )
    print("Raw response:", out)
