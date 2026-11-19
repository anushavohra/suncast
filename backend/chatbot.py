"""Solar chatbot for the /chat endpoint.

Answers with the Gemini API using our FAQ (solar_faq.md) and the city's retailers.
If there is no API key or the API fails, it falls back to the best matching FAQ section.
The API key is read from the environment variable GEMINI_API_KEY (never put it in Git).
"""

import os
import re
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
MAX_MESSAGE_CHARS = 1000

SYSTEM_PROMPT = (
    "You are SunCast's solar assistant for people in Pakistan. "
    "Answer questions about rooftop solar (inverters, batteries, net metering and net billing, "
    "cleaning, choosing a retailer) in simple, short answers (under 120 words). "
    "Reply in the same language the user writes in (English, Urdu or Roman Urdu). "
    "Use only the FAQ and the retailer list below for facts. "
    "If the answer is not there or you are not sure, say so and suggest checking with the "
    "electricity company (DISCO), NEPRA, or a qualified installer. "
    "Rules and rates change, so tell users to confirm current net billing rates. "
    "Never invent prices, retailers or guarantees of savings."
)


def load_faq():
    with open(os.path.join(HERE, "solar_faq.md"), encoding="utf-8") as f:
        return f.read()


def load_retailers(city, limit=5):
    """Top retailers of a city as plain text lines (empty string if none)."""
    for path in (os.path.join(HERE, "data", "retailers.csv"), os.path.join(HERE, "retailers.csv")):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        if "city" in df.columns and city:
            df = df[df["city"].astype(str).str.lower() == city.lower()]
        if "rating" in df.columns:
            df = df.assign(_r=pd.to_numeric(df["rating"], errors="coerce"))
            df = df.sort_values("_r", ascending=False)
        lines = []
        for _, row in df.head(limit).iterrows():
            lines.append("- {} ({}), contact {}, rating {}".format(
                row.get("name", ""), row.get("area", ""),
                row.get("contact", ""), row.get("rating", "")))
        return "\n".join(lines)
    return ""


def faq_fallback(message):
    """Return the FAQ section that shares the most words with the question."""
    sections = [s for s in re.split(r"\n(?=## )", load_faq()) if s.startswith("## ")]
    words = set(re.findall(r"[a-z]{3,}", message.lower()))
    best, best_score = None, 0
    for sec in sections:
        heading = sec.split("\n")[0]
        score = len(words & set(re.findall(r"[a-z]{3,}", sec.lower())))
        score += 3 * len(words & set(re.findall(r"[a-z]{3,}", heading.lower())))
        if score > best_score:
            best, best_score = sec, score
    if best is None:
        return ("I am not sure about that. Please ask about inverters, batteries, "
                "net billing, panel cleaning or choosing a retailer.")
    return best.strip()


def _answer(message, city=""):
    """Reply to one user message (each message is answered on its own)."""
    message = (message or "").strip()[:MAX_MESSAGE_CHARS]
    if not message:
        return "Please type a question about rooftop solar."
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return faq_fallback(message)

    context = "FAQ:\n" + load_faq()
    retailers = load_retailers(city)
    if retailers:
        context += "\n\nRetailers in {}:\n{}".format(city, retailers)

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT + "\n\n" + context}]},
        "contents": [{"role": "user", "parts": [{"text": message}]}],
    }
    try:
        r = requests.post(
            API_URL.format(MODEL),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:  # network error, quota, wrong model name, ...
        print("Gemini call failed:", exc)
        return faq_fallback(message)
    
def answer(message, city=""):
    """Never crashes: returns a friendly message if anything goes wrong."""
    try:
        return _answer(message, city)
    except Exception as exc:
        print("Chatbot error:", exc)
        return "Sorry, I could not answer that right now. Please try again in a moment."





if __name__ == "__main__":
    for q in ["Is a battery worth it?", "What is net billing?", "How do I clean my panels?"]:
        print("Q:", q)
        print("A:", answer(q, "Lahore"))
        print()