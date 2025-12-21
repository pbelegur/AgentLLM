import feedparser
import requests
import json
import os
import re
from datetime import datetime, timezone, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter

# =========================================================
# CONFIG
# =========================================================

TOP_K_OVERALL = 30          # show more than 5
PER_FEED_LIMIT = 25         # pull more per source
HN_LIMIT = 50               # more supply

MAX_PER_SOURCE = 6          # diversity cap
HISTORY_DAYS = 7            # trends window

ENABLE_LLM = False          # placeholder; still rule-based for speed

RSS_FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "WIRED (AI)": "https://www.wired.com/feed/tag/ai/latest/rss",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/gadgets",
    "Gizmodo": "https://gizmodo.com/rss",
    "CNET": "https://www.cnet.com/rss/news/",
    "Engadget": "https://www.engadget.com/rss.xml",
    "Mashable (Tech)": "https://mashable.com/feeds/rss/tech",
}

# What counts as "Builder Radar"
BUILDER_TAGS = {
    "MODEL_RELEASE",
    "SDK_CHANGE",
    "OSS_RELEASE",
    "DEPRECATION",
    "CVE_SECURITY",
    "INFRA",
    "PRICING",
    "OUTAGE",
    "RESEARCH",
}

# What counts as "Product Watch"
PRODUCT_TAGS = {
    "PRODUCT_FEATURE",
    "PRICING",
    "OUTAGE",
}

# =========================================================
# FILESYSTEM
# =========================================================

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/history", exist_ok=True)


def save_json(path: str, obj):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =========================================================
# INGESTION
# =========================================================

def _fetch_rss_one(source_name: str, url: str, limit=PER_FEED_LIMIT, timeout=8):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        out = []
        for e in feed.entries[:limit]:
            title = getattr(e, "title", "") or ""
            title = title.strip()
            link = getattr(e, "link", "") or getattr(e, "id", "") or ""
            link = link.strip()
            if not title:
                continue
            out.append({"title": title, "url": link, "source": source_name})
        return out
    except Exception:
        return []


def fetch_all_rss():
    items = []
    with ThreadPoolExecutor(max_workers=min(12, len(RSS_FEEDS))) as ex:
        futs = [ex.submit(_fetch_rss_one, name, url) for name, url in RSS_FEEDS.items()]
        for fut in as_completed(futs):
            items.extend(fut.result())
    return items


def fetch_hackernews(limit=HN_LIMIT, timeout=8):
    try:
        ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=timeout
        ).json()[:limit]
    except Exception:
        return []

    def fetch_one(sid):
        try:
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                timeout=timeout
            ).json()
            if item and "title" in item:
                url = item.get("url") or f"https://news.ycombinator.com/item?id={sid}"
                return {"title": item["title"], "url": url, "source": "HackerNews"}
        except Exception:
            return None
        return None

    out = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch_one, sid) for sid in ids]
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                out.append(res)
    return out


def dedupe_items(items):
    seen = set()
    out = []
    for it in items:
        key = (it.get("url") or it.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# =========================================================
# MEMORY + HISTORY
# =========================================================

def load_signal_memory(path="data/signal_memory.json"):
    return load_json(path, default={})


def save_signal_memory(memory, path="data/signal_memory.json"):
    save_json(path, memory)


def update_signal_memory(item, memory):
    key = (item.get("url") or item.get("title") or "").strip()
    if not key:
        return memory
    now = datetime.utcnow().isoformat()
    prev = memory.get(key)
    if not prev:
        memory[key] = {
            "first_seen": now,
            "last_seen": now,
            "count": 1,
            "tag": item.get("tag", "OTHER"),
        }
    else:
        prev["last_seen"] = now
        prev["count"] = int(prev.get("count", 0)) + 1
        prev["tag"] = item.get("tag", prev.get("tag", "OTHER"))
    return memory


def save_daily_history(signals):
    # Save timestamped snapshots so trends move during the day
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H00")
    path = f"data/history/{stamp}.json"

    compact = []
    for s in signals:
        compact.append({
            "title": s.get("title"),
            "url": s.get("url"),
            "source": s.get("source"),
            "tag": s.get("tag"),
            "topics": s.get("topics", []),
            "priority": s.get("priority"),
            "score": s.get("score", 0),
        })

    save_json(path, {"date": stamp, "signals": compact})



def load_history(days=HISTORY_DAYS):
    ensure_dirs()
    files = [f for f in os.listdir("data/history") if f.endswith(".json")]
    files.sort()  # timestamped names will sort correctly
    hist = []
    for fname in files[-days:]:
        hist.append(load_json(os.path.join("data/history", fname), default=None))
    return [h for h in hist if h and "signals" in h]



# =========================================================
# AGENTS (RULE-BASED)
# =========================================================

class AIRelevanceAgent:
    """
    Tight relevance: avoids false positives like 'Steam Deck model'.
    """
    AI_WORD_RE = re.compile(r"\bai\b", re.IGNORECASE)

    STRONG_ENTITIES = [
        "openai", "anthropic", "deepmind", "google", "gemini", "microsoft",
        "nvidia", "cuda", "chatgpt", "gpt", "claude", "llama", "ollama",
        "hugging face", "transformers", "pytorch", "tensorflow", "vllm", "triton",
        "langchain", "llamaindex", "milvus", "pinecone", "weaviate"
    ]

    AI_PHRASES = [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "large language model",
        "language model",
        "foundation model",
        "generative ai",
        "diffusion",
        "computer vision",
        "inference",
        "fine-tune",
        "fine tuning",
        "rlhf",
        "rag",
        "vector database",
        "embedding",
        "prompt injection",
        "hallucination",
    ]

    NOISE = [
        "deal", "sale", "discount", "coupon", "gift", "review", "best ", "hands-on",
        "iphone case", "headphones", "tv deal"
    ]

    def is_relevant(self, item):
        title = (item.get("title") or "").strip()
        t = title.lower()

        has_noise = any(n in t for n in self.NOISE)
        strong = any(e in t for e in self.STRONG_ENTITIES)

        matched = []
        if self.AI_WORD_RE.search(title):
            matched.append("ai")
        for p in self.AI_PHRASES:
            if p in t:
                matched.append(p)

        decision = (strong or bool(matched)) and (strong or not has_noise)

        item["decision_trace"] = {
            "agent": "AIRelevanceAgent",
            "strong_entity": strong,
            "matched_phrases": matched[:8],
            "has_noise": has_noise,
            "decision": decision
        }
        return decision


class EntityAgent:
    """
    Extract rough entities/tools from titles.
    """
    ENTITIES = [
        "OpenAI", "ChatGPT", "Anthropic", "Claude", "Google", "Gemini",
        "DeepMind", "Microsoft", "Azure", "AWS", "NVIDIA", "CUDA",
        "PyTorch", "TensorFlow", "Hugging Face", "Transformers",
        "vLLM", "Triton", "ONNX", "Kubernetes", "Docker",
        "LangChain", "LlamaIndex", "Ollama", "Llama",
        "Redis", "Postgres", "MongoDB"
    ]

    def extract(self, item):
        t = (item.get("title") or "")
        tl = t.lower()
        found = []
        for e in self.ENTITIES:
            if e.lower() in tl:
                found.append(e)
        item["entities"] = found
        return found


class TaggingAgent:
    """
    Turn headlines into "what kind of change" tags.
    """
    def tag(self, item):
        t = (item.get("title") or "").lower()

        # Security
        if any(k in t for k in ["cve-", "vulnerability", "security flaw", "exploit", "patched", "zero-day"]):
            tag = "CVE_SECURITY"

        # Outages / incidents
        elif any(k in t for k in ["outage", "down", "incident", "service disruption", "degraded", "postmortem"]):
            tag = "OUTAGE"

        # Deprecation / breaking changes
        elif any(k in t for k in ["deprecated", "deprecation", "sunset", "end of life", "eol", "breaking change"]):
            tag = "DEPRECATION"

        # Pricing / limits
        elif any(k in t for k in ["pricing", "price", "cost", "subscription", "rate limit", "quota"]):
            tag = "PRICING"

        # Hiring signal
        elif any(k in t for k in ["hiring", "job", "jobs", "recruit", "headcount", "layoff", "layoffs"]):
            tag = "HIRING"

        # Infra
        elif any(k in t for k in ["gpu", "cuda", "chip", "h100", "b200", "inference server", "datacenter", "cluster", "kubernetes"]):
            tag = "INFRA"

        # OSS releases
        elif any(k in t for k in ["open source", "open-source", "github", "released", "release", "v1.", "v2.", "v3.", "rc"]):
            tag = "OSS_RELEASE"

        # SDK/API changes
        elif any(k in t for k in ["api", "sdk", "endpoint", "developer", "docs", "tooling", "client library"]):
            tag = "SDK_CHANGE"

        # Model releases / capabilities
        elif any(k in t for k in ["llm", "language model", "foundation model", "gpt", "claude", "gemini", "llama", "chatgpt", "new model"]):
            tag = "MODEL_RELEASE"

        # Product features
        elif any(k in t for k in ["feature", "rollout", "launch", "adds", "new", "update", "introduces", "lets users", "now allows"]):
            tag = "PRODUCT_FEATURE"

        else:
            tag = "OTHER"

        item["tag_trace"] = {"agent": "TaggingAgent", "tag": tag}
        item["tag"] = tag
        return tag


class TopicAgent:
    """
    Assign one or more topics for trends + clustering.
    """
    TOPIC_RULES = {
        "models": ["gpt", "claude", "gemini", "llama", "chatgpt", "model", "llm", "foundation model", "language model"],
        "agents": ["agent", "multi-agent", "tool use", "function calling", "autonomous"],
        "inference": ["inference", "serving", "latency", "throughput", "vllm", "triton", "onnx"],
        "training": ["training", "fine-tune", "finetune", "rlhf", "dataset"],
        "gpu": ["gpu", "cuda", "nvidia", "h100", "b200", "chip"],
        "security": ["cve", "vulnerability", "exploit", "prompt injection", "jailbreak"],
        "policy": ["law", "lawmakers", "regulation", "ban", "safety", "rules", "standards", "privacy"],
        "open_source": ["open-source", "open source", "github", "released", "release"],
        "devtools": ["sdk", "api", "endpoint", "developer", "docs", "library"],
        "products": ["feature", "rollout", "launch", "update", "introduces", "lets users"],
        "hiring": ["hiring", "jobs", "recruit", "layoffs"],
    }

    def topics(self, item):
        t = (item.get("title") or "").lower()
        out = []
        for topic, kws in self.TOPIC_RULES.items():
            if any(k in t for k in kws):
                out.append(topic)
        # keep stable order
        item["topics"] = sorted(set(out))
        return item["topics"]


class PriorityAgent:
    """
    Convert tag+entities into a priority + numeric score.
    """
    TAG_WEIGHT = {
        "CVE_SECURITY": 10,
        "OUTAGE": 9,
        "DEPRECATION": 8,
        "PRICING": 7,
        "SDK_CHANGE": 7,
        "MODEL_RELEASE": 7,
        "INFRA": 7,
        "OSS_RELEASE": 6,
        "HIRING": 4,
        "PRODUCT_FEATURE": 4,
        "RESEARCH": 5,
        "OTHER": 1,
    }

    ENTITY_BOOST = {
        "OpenAI": 3, "ChatGPT": 2, "Anthropic": 3, "Claude": 2, "Google": 2, "Gemini": 2,
        "NVIDIA": 3, "CUDA": 3, "Hugging Face": 2, "Transformers": 2, "vLLM": 3, "Triton": 3,
    }

    def score(self, item):
        tag = item.get("tag", "OTHER")
        score = self.TAG_WEIGHT.get(tag, 1)

        for e in item.get("entities", []):
            score += self.ENTITY_BOOST.get(e, 0)

        # small boost for multiple topics (usually richer)
        score += min(len(item.get("topics", [])), 3)

        item["score"] = score

        if score >= 12:
            pr = "HIGH"
        elif score >= 7:
            pr = "MED"
        else:
            pr = "LOW"

        item["priority"] = pr
        return pr, score


class TemporalChangeAgent:
    def analyze(self, item, memory):
        key = (item.get("url") or item.get("title") or "").strip()
        prev = memory.get(key)
        if not prev:
            decision = "NEW"
            prev_count = 0
        else:
            prev_count = int(prev.get("count", 0))
            decision = "ESCALATING" if prev_count >= 2 else "ONGOING"

        item["temporal_trace"] = {
            "agent": "TemporalChangeAgent",
            "previous_count": prev_count,
            "decision": decision
        }
        item["change_type"] = decision
        return decision


class ActionAgent:
    """
    Make the output *do something*.
    """
    def action(self, item):
        tag = item.get("tag", "OTHER")
        topics = item.get("topics", [])
        entities = item.get("entities", [])

        # actions are short + concrete
        if tag == "CVE_SECURITY":
            act = "Check if you use the affected library/service; patch/upgrade and scan dependencies."
        elif tag == "OUTAGE":
            act = "If you rely on this service, add a fallback plan; note incident learnings for resilience."
        elif tag == "DEPRECATION":
            act = "Find the replacement and pin/upgrade before the sunset date; update docs/tests."
        elif tag == "PRICING":
            act = "Recalculate cost impact; consider cheaper tiers/alternatives; update budgets/limits."
        elif tag == "SDK_CHANGE":
            act = "Skim changelog/docs; verify your calls still work; update client version if needed."
        elif tag == "OSS_RELEASE":
            act = "Skim release notes; if it matches your stack, backlog a quick benchmark or POC."
        elif tag == "MODEL_RELEASE":
            act = "Check capabilities/limits; decide if it improves your use-case; run a small eval."
        elif tag == "INFRA":
            act = "If this impacts inference/training cost, note it; consider benchmarking/architecture tweaks."
        elif tag == "PRODUCT_FEATURE":
            act = "If you use this product, test the feature; note if it changes your workflow or requirements."
        elif tag == "HIRING":
            act = "Optional: just track it as a market signal (no action needed)."
        else:
            act = "Track if it becomes relevant; no immediate action."

        # tiny personalization by topic
        if "security" in topics and tag != "CVE_SECURITY":
            act = "Security-relevant: verify configs, permissions, and safe prompt/tooling patterns."

        item["action"] = act
        return act


# =========================================================
# PIPELINE HELPERS
# =========================================================

def add_drop_trace(item, agent, reason, details=None):
    item["drop_trace"] = {
        "agent": agent,
        "reason": reason,
        "details": details or {}
    }
    return item


def diversify_and_select(items, k=TOP_K_OVERALL):
    # sort by score desc, then diverse by source cap
    ranked = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    chosen = []
    per_source = Counter()
    for it in ranked:
        src = it.get("source", "Unknown")
        if per_source[src] >= MAX_PER_SOURCE:
            continue
        chosen.append(it)
        per_source[src] += 1
        if len(chosen) >= k:
            break
    return chosen


def compute_topic_trends(history):
    """
    history: list of {"date": ..., "signals":[...]}
    returns: topics list with counts + delta vs yesterday
    """
    if not history:
        return []

    # last day and previous day
    last = history[-1]["signals"]
    prev = history[-2]["signals"] if len(history) >= 2 else []

    def topic_counts(signals):
        c = Counter()
        for s in signals:
            for t in s.get("topics", []) or []:
                c[t] += 1
        return c

    last_c = topic_counts(last)
    prev_c = topic_counts(prev)

    # 7-day counts
    week_c = Counter()
    for day in history:
        for s in day["signals"]:
            for t in s.get("topics", []) or []:
                week_c[t] += 1

    topics = []
    for t, total in week_c.most_common():
        topics.append({
            "topic": t,
            "count_7d": total,
            "count_today": last_c.get(t, 0),
            "delta_vs_yesterday": last_c.get(t, 0) - prev_c.get(t, 0),
        })
    return topics


def new_since_yesterday(history, today_signals):
    if len(history) < 2:
        return today_signals[:]

    yesterday = history[-2]["signals"]
    y_keys = set((s.get("url") or s.get("title") or "").strip().lower() for s in yesterday)
    out = []
    for s in today_signals:
        key = (s.get("url") or s.get("title") or "").strip().lower()
        if key and key not in y_keys:
            out.append(s)
    return out


def generate_brief(signals):
    if not signals:
        return "Good morning.\n\nNo meaningful AI signals detected."

    lines = ["Good morning.\n", "Here are the AI signals that matter today:\n"]
    for i, s in enumerate(signals[:10], 1):
        lines.append(
            f"{i}. {s['title']}\n"
            f"   Tag: {s.get('tag')} • Priority: {s.get('priority')} • Change: {s.get('change_type')} • Source: {s.get('source')}\n"
            f"   Action: {s.get('action')}\n"
        )
    return "\n".join(lines)


def atomic_save_json(path: str, obj):
    """
    Write JSON atomically so the UI never reads a half-written file.
    """
    ensure_dirs()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def save_latest_snapshot(payload: dict):
    payload = dict(payload)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_save_json("data/latest.json", payload)



# =========================================================
# MAIN ORCHESTRATOR
# =========================================================

def run_pipeline():
    ensure_dirs()

    # ingest
    rss_items = fetch_all_rss()
    hn_items = fetch_hackernews()
    all_news = dedupe_items(rss_items + hn_items)
    save_json("data/raw_news.json", all_news)

    memory = load_signal_memory()

    relevance = AIRelevanceAgent()
    entity_agent = EntityAgent()
    tagger = TaggingAgent()
    topic_agent = TopicAgent()
    priority_agent = PriorityAgent()
    temporal = TemporalChangeAgent()
    action_agent = ActionAgent()

    dropped = []
    kept = []

    # 1) relevance filter
    for it in all_news:
        if relevance.is_relevant(it):
            kept.append(it)
        else:
            dropped.append(add_drop_trace(it, "AIRelevanceAgent", "Not AI-relevant", it.get("decision_trace")))

    # 2) annotate
    for it in kept:
        entity_agent.extract(it)
        tagger.tag(it)
        topic_agent.topics(it)
        priority_agent.score(it)
        temporal.analyze(it, memory)
        action_agent.action(it)

    # 3) select overall + diversity
    selected = diversify_and_select(kept, k=TOP_K_OVERALL)

    # 4) update memory based on selected only (keeps “novelty” meaningful)
    for it in selected:
        update_signal_memory(it, memory)
    save_signal_memory(memory)

    # 5) history + trends
    save_daily_history(selected)
    hist = load_history(HISTORY_DAYS)
    topics = compute_topic_trends(hist)
    new_items = new_since_yesterday(hist, selected)

    # 6) views
    builder_radar = [s for s in selected if s.get("tag") in BUILDER_TAGS]
    product_watch = [s for s in selected if s.get("tag") in PRODUCT_TAGS]

    # Action queue = sorted by priority then score
    pr_order = {"HIGH": 0, "MED": 1, "LOW": 2}
    action_queue = sorted(
        selected,
        key=lambda x: (pr_order.get(x.get("priority", "LOW"), 9), -x.get("score", 0))
    )

    # brief
    brief = generate_brief(selected)

        # Save a "latest" snapshot for the UI to read quickly
    save_latest_snapshot({
        "brief": brief,
        "signals": selected,
        "builder_radar": builder_radar,
        "product_watch": product_watch,
        "action_queue": action_queue[:25],
        "topic_trends": topics[:25],
        "new_since_yesterday": new_items[:25],
        "stats": {
            "raw_total": len(all_news),
            "kept_after_relevance": len(kept),
            "selected": len(selected),
            "builder_radar": len(builder_radar),
            "product_watch": len(product_watch),
            "dropped": len(dropped),
        }
    })



    return {
        "brief": brief,
        "signals": selected,                 # overall
        "builder_radar": builder_radar,
        "product_watch": product_watch,
        "action_queue": action_queue[:25],
        "topic_trends": topics[:25],
        "new_since_yesterday": new_items[:25],
        "dropped": dropped,
        "stats": {
            "raw_total": len(all_news),
            "kept_after_relevance": len(kept),
            "selected": len(selected),
            "builder_radar": len(builder_radar),
            "product_watch": len(product_watch),
            "dropped": len(dropped),
        }
    }


if __name__ == "__main__":
    out = run_pipeline()
    print(out["brief"])
