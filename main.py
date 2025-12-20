import feedparser
import requests
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ENABLE_LLM = False      # switch to True when needed
ENABLE_AUDIO = False    # keep False for speed


# =========================================================
# UTILS / FILESYSTEM
# =========================================================

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)


def save_news_to_file(news, path):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(news, f, indent=2)


def add_drop_trace(item, agent, reason, details=None):
    item["drop_trace"] = {
        "agent": agent,
        "reason": reason,
        "details": details or {}
    }
    return item


def score_news(items):
    priority = ["openai", "nvidia", "model", "llm", "hiring", "interview", "compute"]
    return sorted(
        [(sum(1 for k in priority if k in i["title"].lower()), i) for i in items],
        key=lambda x: x[0],
        reverse=True
    )


def generate_daily_brief(items):
    if not items:
        return "Good morning.\n\nNo high-impact AI signals detected."

    lines = ["Good morning.\n", "Here are the AI signals that actually matter today:\n"]

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
# SIGNAL MEMORY (FIXED: was missing in your file)
# =========================================================

def load_signal_memory(path="data/signal_memory.json"):
    ensure_dirs()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_signal_memory(memory, path="data/signal_memory.json"):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def update_signal_memory(item, memory):
    title = item["title"]
    today = datetime.now().strftime("%Y-%m-%d")

    if title not in memory or not isinstance(memory[title], dict):
        memory[title] = {
            "signal_type": item.get("signal_type", "GENERAL_AI"),
            "first_seen": today,
            "last_seen": today,
            "count": 1
        }
        return memory

    rec = memory[title]

    # If legacy record missing fields, repair without inflating count
    if "first_seen" not in rec:
        rec["first_seen"] = today
    if "last_seen" not in rec:
        rec["last_seen"] = today
        return memory
    if "count" not in rec:
        rec["count"] = 1
        rec["last_seen"] = today
        return memory

    # Normal behavior: increment at most once per day
    if rec["last_seen"] != today:
        rec["count"] += 1
        rec["last_seen"] = today

    return memory



def generate_audio_from_text(text, path):
    """
    Lazy import so audio libs don't slow down normal runs.
    """
    import pyttsx3
    ensure_dirs()
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()


# =========================================================
# AGENTS
# =========================================================

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

        matched_ai = [k for k in ai_keywords if k in title]
        matched_companies = [c for c in companies if c in title]

        decision = bool(matched_ai or matched_companies)

        item["decision_trace"] = {
            "agent": "AIRelevanceAgent",
            "matched_ai_keywords": matched_ai,
            "matched_companies": matched_companies,
            "decision": decision
        }

        return decision


class SignalClassificationAgent:
    """
    What KIND of AI signal is this?
    """
    def classify(self, item):
        title = item["title"].lower()
        matched = []

        if any(k in title for k in ["openai", "llm", "model", "foundation"]):
            signal = "FOUNDATION_MODEL"
            matched = ["openai/llm/model"]

        elif any(k in title for k in ["nvidia", "gpu", "compute", "infrastructure"]):
            signal = "AI_INFRASTRUCTURE"
            matched = ["nvidia/gpu/compute"]

        elif any(k in title for k in ["hiring", "interview", "recruiting"]):
            signal = "HIRING_SIGNAL"
            matched = ["hiring/interview"]

        elif any(k in title for k in ["policy", "regulation", "law", "judge"]):
            signal = "AI_POLICY"
            matched = ["policy/regulation"]

        elif any(k in title for k in ["app", "feature", "consumer", "glasses"]):
            signal = "CONSUMER_AI"
            matched = ["consumer features"]

        else:
            signal = "GENERAL_AI"

        item["classification_trace"] = {
            "agent": "SignalClassificationAgent",
            "matched_patterns": matched,
            "output": signal
        }

        return signal


class TemporalChangeAgent:
    """
    NEW / ONGOING / ESCALATING based on signal memory
    """
    def analyze(self, item, signal_memory):
        title = item["title"]
        record = signal_memory.get(title)

        if record is None:
            change = "NEW"
            count = 0
        else:
            count = record.get("count", 0)
            if count < 2:
                change = "ONGOING"
            else:
                change = "ESCALATING"

        item["temporal_trace"] = {
            "agent": "TemporalChangeAgent",
            "previous_count": count,
            "decision": change
        }

        return change


class ImpactAnalysisAgent:
    """
    Impact ONLY for important signal types.
    """
    def analyze(self, item):
        title = item["title"].lower()
        signal_type = item.get("signal_type", "GENERAL_AI")

        roles = []
        skills = []
        interview_relevance = "LOW"
        why = "General AI industry update."

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
    Lazy import so it doesn't slow down fast mode.
    """
    def __init__(self):
        from transformers import pipeline
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=128
        )

    def synthesize(self, item):
        prompt = (
            f"Explain why the following AI development matters for engineers "
            f"in the next 3 to 6 months.\n\n"
            f"Title: {item['title']}\n"
            f"Signal type: {item['signal_type']}\n"
            f"Impact summary: {item['impact']['why']}\n"
        )
        return self.generator(prompt)[0]["generated_text"].strip()


# =========================================================
# INGESTION
# =========================================================

def fetch_techcrunch():
    feed = feedparser.parse("https://techcrunch.com/feed/")
    return [{"title": e.title, "url": e.link, "source": "TechCrunch"} for e in feed.entries[:15]]


def fetch_hackernews(limit=15, timeout=6):
    ids = requests.get(
        "https://hacker-news.firebaseio.com/v0/topstories.json",
        timeout=timeout
    ).json()[:limit]

    def fetch_one(sid):
        item = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
            timeout=timeout
        ).json()
        if item and "title" in item:
            return {
                "title": item["title"],
                "url": item.get("url", ""),
                "source": "HackerNews"
            }
        return None

    articles = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(fetch_one, sid) for sid in ids]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                articles.append(res)

    return articles


# =========================================================
# ORCHESTRATOR
# =========================================================

class AgentOrchestrator:
    def __init__(self):
        self.relevance = AIRelevanceAgent()
        self.classifier = SignalClassificationAgent()
        self.temporal = TemporalChangeAgent()
        self.impact = ImpactAnalysisAgent()
        self.synthesizer = SynthesisAgent() if ENABLE_LLM else None

    def run(self, news_items, signal_memory):
        dropped = []

        # Relevance
        ai_news = []
        for n in news_items:
            if self.relevance.is_relevant(n):
                ai_news.append(n)
            else:
                dropped.append(add_drop_trace(
                    n,
                    "AIRelevanceAgent",
                    "Not AI-relevant",
                    {"decision_trace": n.get("decision_trace", {})}
                ))

        # Classification
        for item in ai_news:
            item["signal_type"] = self.classifier.classify(item)

        # Filter (keep only decision-relevant categories)
        filtered = []
        for n in ai_news:
            if n["signal_type"] in {
                "FOUNDATION_MODEL",
                "AI_INFRASTRUCTURE",
                "HIRING_SIGNAL",
                "AI_POLICY"
            }:
                filtered.append(n)
            else:
                dropped.append(add_drop_trace(
                    n,
                    "SignalClassificationAgent",
                    "Filtered low-value category",
                    {"signal_type": n["signal_type"], "classification_trace": n.get("classification_trace", {})}
                ))

        # Scoring
        scored = score_news(filtered)
        ranked = [item for _, item in scored]

        top_news = ranked[:5]
        for n in ranked[5:]:
            dropped.append(add_drop_trace(
                n,
                "ScoringAgent",
                "Not in top K after scoring",
                {"k": 5}
            ))

        # Temporal + Impact
        for item in top_news:
            item["change_type"] = self.temporal.analyze(item, signal_memory)
            item["impact"] = self.impact.analyze(item)

        # Optional LLM
        for item in top_news:
            if self.synthesizer:
                item["llm_reasoning"] = self.synthesizer.synthesize(item)
            else:
                item["llm_reasoning"] = "LLM synthesis disabled (fast mode)."

        return top_news, dropped


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline():
    ensure_dirs()

    tech = fetch_techcrunch()
    hn = fetch_hackernews()
    all_news = tech + hn

    signal_memory = load_signal_memory()

    orchestrator = AgentOrchestrator()
    top_news, dropped = orchestrator.run(all_news, signal_memory)

    # Update memory based on kept signals (once per day)
    for item in top_news:
        signal_memory = update_signal_memory(item, signal_memory)
    save_signal_memory(signal_memory)

    # Save raw news separately
    save_news_to_file(all_news, "data/raw_news.json")

    brief = generate_daily_brief(top_news)

    # Audio only if enabled
    if ENABLE_AUDIO:
        try:
            generate_audio_from_text(brief, "output/daily_brief.wav")
        except Exception:
            pass

    return {
        "brief": brief,
        "signals": top_news,
        "dropped": dropped
    }


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    result = run_pipeline()
    print(result["brief"])
