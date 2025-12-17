import feedparser
import requests
import json
import os
import pyttsx3
from transformers import pipeline


class AIRelevanceAgent:
    """
    Coarse filter: is this even AI-related?
    """
    def is_relevant(self, item):
        title = item["title"].lower()

        ai_keywords = [
            "ai", "artificial intelligence", "llm", "model",
            "machine learning", "deep learning", "neural",
            "inference", "training", "autopilot", "computer vision"
        ]

        companies = [
            "nvidia", "openai", "google", "microsoft",
            "amazon", "meta", "tesla"
        ]

        return (
            any(k in title for k in ai_keywords) or
            any(c in title for c in companies)
        )


class SignalClassificationAgent:
    """
    What KIND of AI signal is this?
    """
    def classify(self, item):
        title = item["title"].lower()

        if any(k in title for k in ["openai", "llm", "model", "foundation"]):
            return "FOUNDATION_MODEL"

        if any(k in title for k in ["nvidia", "gpu", "compute", "infrastructure"]):
            return "AI_INFRASTRUCTURE"

        if any(k in title for k in ["hiring", "interview", "recruiting"]):
            return "HIRING_SIGNAL"

        if any(k in title for k in ["policy", "regulation", "law", "judge"]):
            return "AI_POLICY"

        if any(k in title for k in ["app", "feature", "consumer", "glasses"]):
            return "CONSUMER_AI"

        return "GENERAL_AI"


class TemporalChangeAgent:
    """
    NEW / ONGOING / ESCALATING
    """
    def analyze(self, item, previous_items):
        title = item["title"].lower()
        prev_titles = [p["title"].lower() for p in previous_items]

        if not prev_titles:
            return "NEW"

        similar = sum(1 for t in prev_titles if title in t or t in title)

        if similar == 0:
            return "NEW"
        elif similar < 2:
            return "ONGOING"
        else:
            return "ESCALATING"


class ImpactAnalysisAgent:
    """
    Impact ONLY for important signal types.
    """
    def analyze(self, item):
        title = item["title"].lower()
        signal_type = item.get("signal_type", "GENERAL_AI")

        # Default conservative output
        roles = []
        skills = []
        interview_relevance = "LOW"
        why = "General AI industry update."

        # Only fire for meaningful signal types
        if signal_type in {"FOUNDATION_MODEL", "AI_INFRASTRUCTURE", "HIRING_SIGNAL"}:

            if "openai" in title or "model" in title:
                roles += ["ML Engineer", "Applied Scientist"]
                skills += ["LLMs", "Model Evaluation", "Prompt Engineering"]
                interview_relevance = "HIGH"
                why = "Indicates changes in foundation model capabilities."

            if "nvidia" in title:
                roles += ["ML Engineer", "Infrastructure Engineer"]
                skills += ["CUDA", "GPU Computing", "LLM Inference"]
                interview_relevance = "HIGH"
                why = "Signals expansion of AI infrastructure and compute demand."

            if "interview" in title or "hiring" in title:
                roles += ["Software Engineer", "ML Engineer"]
                skills += ["System Design", "ML Integration"]
                interview_relevance = "HIGH"
                why = "Directly affects hiring and interview expectations."

        return {
            "affected_roles": list(set(roles)),
            "skills": list(set(skills)),
            "interview_relevance": interview_relevance,
            "why": why
        }

class SynthesisAgent:
    """
    Uses an LLM to explain why a signal matters.
    """

    def __init__(self):
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=128
        )

    def synthesize(self, item):
        prompt = (
            f"Explain why the following AI development matters for engineers and companies "
            f"in the next 3 to 6 months.\n\n"
            f"Title: {item['title']}\n"
            f"Signal type: {item['signal_type']}\n"
            f"Impact summary: {item['impact']['why']}\n"
        )

        result = self.generator(prompt)[0]["generated_text"]
        return result.strip()

# =========================================================
# INGESTION
# =========================================================

def fetch_techcrunch():
    feed = feedparser.parse("https://techcrunch.com/feed/")
    return [
        {"title": e.title, "url": e.link, "source": "TechCrunch"}
        for e in feed.entries[:15]
    ]


def fetch_hackernews():
    ids = requests.get(
        "https://hacker-news.firebaseio.com/v0/topstories.json"
    ).json()[:15]

    articles = []
    for sid in ids:
        item = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
        ).json()
        if item and "title" in item:
            articles.append({
                "title": item["title"],
                "url": item.get("url", ""),
                "source": "HackerNews"
            })
    return articles


# =========================================================
# MEMORY + UTILS
# =========================================================

def load_previous_news(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_news_to_file(news, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(news, f, indent=2)


def detect_new_news(current, previous):
    prev_titles = {p["title"] for p in previous}
    return [c for c in current if c["title"] not in prev_titles]


def score_news(items):
    priority = [
        "openai", "nvidia", "model", "llm",
        "hiring", "interview", "compute"
    ]
    scored = []
    for item in items:
        score = sum(1 for k in priority if k in item["title"].lower())
        scored.append((score, item))
    return sorted(scored, key=lambda x: x[0], reverse=True)


def save_brief_to_file(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def generate_audio_from_text(text, path):
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def generate_daily_brief(items):
    if not items:
        return "Good morning.\n\nNo high-impact AI signals detected."

    lines = [
        "Good morning.\n",
        "Here are the AI signals that actually matter today:\n"
    ]

    for i, item in enumerate(items, 1):
        impact = item["impact"]
        lines.append(
            f"{i}. {item['title']}\n"
            f"   Signal type: {item['signal_type']}\n"
            f"   Temporal status: {item['change_type']}\n"
            f"   Source: {item['source']}\n"
            f"   Impact (rule-based): {impact['why']}\n"
            f"   Impact (LLM reasoning): {item.get('llm_reasoning', 'N/A')}\n"
            f"   Roles: {', '.join(impact['affected_roles']) or 'N/A'}\n"
            f"   Skills: {', '.join(impact['skills']) or 'N/A'}\n"
            f"   Interview relevance: {impact['interview_relevance']}\n"
            f"   Link: {item['url']}\n"
        )
    return "\n".join(lines)


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline():
    tech = fetch_techcrunch()
    hn = fetch_hackernews()

    all_news = tech + hn
    previous = load_previous_news("data/previous_news.json")

    new_news = detect_new_news(all_news, previous)
    candidate = new_news if new_news else all_news

    # AGENT 1: Coarse relevance
    relevance_agent = AIRelevanceAgent()
    ai_news = [n for n in candidate if relevance_agent.is_relevant(n)]

    # AGENT 2: Signal classification
    classification_agent = SignalClassificationAgent()
    for item in ai_news:
        item["signal_type"] = classification_agent.classify(item)

    # STEP B: tighten — drop consumer + general AI
    ai_news = [
        n for n in ai_news
        if n["signal_type"] in {
            "FOUNDATION_MODEL",
            "AI_INFRASTRUCTURE",
            "HIRING_SIGNAL",
            "AI_POLICY"
        }
    ]

    # AGENT 3: Scoring
    scored = score_news(ai_news)
    top_news = [item for _, item in scored][:5]

    # AGENT 4: Temporal reasoning
    temporal_agent = TemporalChangeAgent()
    for item in top_news:
        item["change_type"] = temporal_agent.analyze(item, previous)

    # AGENT 5: Impact analysis
    impact_agent = ImpactAnalysisAgent()
    for item in top_news:
        item["impact"] = impact_agent.analyze(item)
    
    # AGENT 6: LLM Synthesis
    synthesis_agent = SynthesisAgent()
    for item in top_news:
        item["llm_reasoning"] = synthesis_agent.synthesize(item)


    brief = generate_daily_brief(top_news)

    save_news_to_file(all_news, "data/previous_news.json")
    save_brief_to_file(brief, "output/daily_brief.txt")
    generate_audio_from_text(brief, "output/daily_brief.wav")

    return make_json_safe({
        "brief": brief,
        "signals": top_news
    })


# =========================================================
# CLI ENTRY
# =========================================================

if __name__ == "__main__":
    result = run_pipeline()
    print(result["brief"])
