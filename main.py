import feedparser
import requests
import json
import os
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter

TOP_K_OVERALL = 30
PER_FEED_LIMIT = 25
HN_LIMIT = 50

MAX_PER_SOURCE = 6
HISTORY_DAYS = 7

CACHE_TTL_SECONDS = 120

ENABLE_DROPPED = True

ENABLE_LLM_ANNOTATOR = True
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_ANNOTATE_TOP_N = 5
LLM_TIMEOUT_SECONDS = 35

RSS_FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "WIRED": "https://www.wired.com/feed/rss",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
    "Gizmodo": "https://gizmodo.com/rss",
    "CNET": "https://www.cnet.com/rss/all/",
    "Engadget": "https://www.engadget.com/rss.xml",
    "Mashable": "https://mashable.com/feeds/rss/all",
}

HN_TOPSTORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

RAW_NEWS_PATH = "data/raw_news.json"
SIGNAL_MEMORY_PATH = "data/signal_memory.json"
LATEST_PATH = "data/latest.json"
HISTORY_DIR = "data/history"
DAILY_DIGEST_PATH = "data/daily_digest.md"

AI_KEYWORDS = [
    "ai", "artificial intelligence", "llm", "gpt", "model", "inference", "training",
    "transformer", "agents", "agentic", "rag", "embedding", "multimodal",
    "cuda", "gpu", "npu", "accelerator", "quantization", "fine-tuning", "finetuning",
    "prompt", "alignment", "eval", "evaluation", "benchmark", "safety",
]

AI_COMPANIES = [
    "openai", "anthropic", "google", "deepmind", "meta", "microsoft", "nvidia",
    "amazon", "aws", "hugging face", "huggingface", "cohere", "stability", "mistral",
    "ollama", "groq", "perplexity", "xai"
]

RULE_TAGS = [
    "MODEL_RELEASE", "SDK_CHANGE", "OSS_RELEASE", "DEPRECATION",
    "CVE_SECURITY", "OUTAGE", "PRICING", "PRODUCT_FEATURE",
    "HIRING", "INFRA", "RESEARCH", "OTHER"
]

TOPIC_SET = [
    "models", "agents", "inference", "training", "gpu", "security", "policy",
    "open_source", "devtools", "products", "hiring"
]


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def atomic_save_json(path: str, obj):
    ensure_dirs()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def save_latest_snapshot(payload: dict):
    payload = dict(payload)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_save_json(LATEST_PATH, payload)


def fetch_rss(feed_name: str, url: str, limit: int):
    parsed = feedparser.parse(url)
    items = []
    for e in parsed.entries[:limit]:
        items.append({
            "title": getattr(e, "title", "").strip(),
            "url": getattr(e, "link", "").strip(),
            "source": feed_name
        })
    return items


def fetch_hn_top(limit: int):
    ids = requests.get(HN_TOPSTORIES_URL, timeout=20).json()[:limit]

    def fetch_item(i):
        j = requests.get(HN_ITEM_URL.format(i), timeout=20).json()
        if not j:
            return None
        title = (j.get("title") or "").strip()
        url = (j.get("url") or "").strip()
        if not url:
            url = f"https://news.ycombinator.com/item?id={i}"
        return {"title": title, "url": url, "source": "HackerNews"}

    out = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(fetch_item, i) for i in ids]
        for fut in as_completed(futs):
            it = fut.result()
            if it and it.get("title"):
                out.append(it)
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


class AIRelevanceAgent:
    def process(self, item: dict):
        t = (item.get("title") or "").lower()
        matched_kw = [k for k in AI_KEYWORDS if k in t]
        matched_co = [c for c in AI_COMPANIES if c in t]

        decision = bool(matched_kw or matched_co)
        item["decision_trace"] = {
            "agent": "AIRelevanceAgent",
            "decision": decision,
            "matched_ai_keywords": matched_kw[:10],
            "matched_companies": matched_co[:10],
        }
        return decision, item


class EntityAgent:
    def process(self, item: dict):
        t = (item.get("title") or "").lower()
        entities = []
        for c in AI_COMPANIES:
            if c in t:
                entities.append(c)
        item["entities"] = sorted(set(entities))
        return item


class TaggingAgent:
    def process(self, item: dict):
        t = (item.get("title") or "").lower()
        tag = "OTHER"

        if any(x in t for x in ["cve", "vulnerability", "security flaw", "zero-day"]) or (
            "exploit" in t and any(x in t for x in ["vulnerability", "security", "hack", "breach", "attack"])
        ):
            tag = "CVE_SECURITY"
        elif any(x in t for x in ["outage", "down", "incident", "service disruption"]):
            tag = "OUTAGE"
        elif any(x in t for x in ["deprecated", "deprecates", "deprecation", "sunset", "end of life", "eol"]):
            tag = "DEPRECATION"
        elif any(x in t for x in ["price", "pricing", "subscription", "cost", "charges", "raises prices"]):
            tag = "PRICING"
        elif any(x in t for x in ["sdk", "api", "release notes", "cli", "library", "python package", "npm"]):
            tag = "SDK_CHANGE"
        elif any(x in t for x in ["open source", "github", "released", "launches", "introduces", "announces"]):
            tag = "OSS_RELEASE"
        elif any(x in t for x in ["model", "llm", "gpt", "weights", "checkpoint", "fine-tune", "finetune"]):
            tag = "MODEL_RELEASE"
        elif any(x in t for x in ["hiring", "jobs", "layoffs", "recruiting"]):
            tag = "HIRING"
        elif any(x in t for x in ["gpu", "cuda", "inference", "datacenter", "cloud", "kubernetes", "server"]):
            tag = "INFRA"
        elif any(x in t for x in ["paper", "arxiv", "research", "benchmark"]):
            tag = "RESEARCH"
        elif any(x in t for x in ["feature", "rollout", "adds", "update", "improves"]):
            tag = "PRODUCT_FEATURE"

        item["tag"] = tag
        item["tag_trace"] = {"agent": "TaggingAgent", "tag": tag}
        return item


class TopicAgent:
    def process(self, item: dict):
        t = (item.get("title") or "").lower()
        topics = set()

        if any(x in t for x in ["model", "llm", "gpt", "weights", "checkpoint"]):
            topics.add("models")
        if any(x in t for x in ["agent", "agentic", "tool use", "function calling"]):
            topics.add("agents")
        if any(x in t for x in ["inference", "latency", "throughput", "quantization", "serving"]):
            topics.add("inference")
        if any(x in t for x in ["training", "fine-tune", "finetune", "rlhf"]):
            topics.add("training")
        if any(x in t for x in ["gpu", "cuda", "nvidia", "accelerator"]):
            topics.add("gpu")
        if any(x in t for x in ["cve", "vulnerability", "security", "exploit"]):
            topics.add("security")
        if any(x in t for x in ["policy", "regulation", "lawmakers", "ban", "standards"]):
            topics.add("policy")
        if any(x in t for x in ["open source", "github", "apache", "mit license"]):
            topics.add("open_source")
        if any(x in t for x in ["sdk", "api", "cli", "library"]):
            topics.add("devtools")
        if any(x in t for x in ["feature", "rollout", "product", "subscription"]):
            topics.add("products")
        if any(x in t for x in ["hiring", "jobs", "recruiting"]):
            topics.add("hiring")

        item["topics"] = sorted(topics)
        return item


class PriorityAgent:
    def process(self, item: dict):
        score = 0
        tag = item.get("tag", "OTHER")
        topics = item.get("topics", [])
        ents = item.get("entities", [])

        if tag in ("CVE_SECURITY", "OUTAGE", "DEPRECATION"):
            score += 60
        elif tag in ("MODEL_RELEASE", "SDK_CHANGE", "PRICING"):
            score += 40
        elif tag in ("OSS_RELEASE", "INFRA"):
            score += 25
        else:
            score += 10

        if "security" in topics:
            score += 20
        if "inference" in topics or "gpu" in topics:
            score += 15
        if "models" in topics:
            score += 10

        boost_entities = {"openai", "anthropic", "nvidia", "microsoft", "google", "deepmind", "meta"}
        score += 5 * len([e for e in ents if e in boost_entities])

        priority = "LOW"
        if score >= 70:
            priority = "HIGH"
        elif score >= 40:
            priority = "MED"

        item["score"] = score
        item["priority"] = priority
        return item


class TemporalChangeAgent:
    def __init__(self, memory: dict):
        self.memory = memory

    def _key(self, item: dict):
        return (item.get("url") or item.get("title") or "").strip().lower()

    def process(self, item: dict):
        k = self._key(item)
        rec = self.memory.get(k)

        change = "NEW"
        if rec:
            prev_count = rec.get("count", 0)
            change = "ONGOING" if prev_count < 3 else "ESCALATING"

        item["change_type"] = change
        item["temporal_trace"] = {"agent": "TemporalChangeAgent", "decision": change, "previous": rec}
        return item

    def update(self, items: list):
        now = datetime.now(timezone.utc).isoformat()
        for it in items:
            k = self._key(it)
            if not k:
                continue
            if k not in self.memory:
                self.memory[k] = {
                    "first_seen": now,
                    "last_seen": now,
                    "count": 1,
                    "source": it.get("source"),
                    "tag": it.get("tag"),
                }
            else:
                self.memory[k]["last_seen"] = now
                self.memory[k]["count"] = int(self.memory[k].get("count", 0)) + 1
                self.memory[k]["tag"] = it.get("tag")


class ActionAgent:
    def process(self, item: dict):
        tag = item.get("tag", "OTHER")
        topics = item.get("topics", [])

        action = "Skim the article and decide if it affects your stack."
        if tag == "CVE_SECURITY":
            action = "Check if your stack uses the affected component; patch/mitigate and add monitoring."
        elif tag == "OUTAGE":
            action = "Review incident details; validate your fallback strategy and status monitoring."
        elif tag == "DEPRECATION":
            action = "Identify impacted dependencies and plan migration before end-of-life."
        elif tag == "SDK_CHANGE":
            action = "Scan release notes; update SDK and run integration tests."
        elif tag == "MODEL_RELEASE":
            action = "Evaluate the model; run a small benchmark on your use-case and compare costs/latency."
        elif tag == "PRICING":
            action = "Recalculate spend impact; evaluate alternatives or adjust usage limits."

        if "inference" in topics:
            action = action + " Focus on latency/throughput implications."

        item["action"] = action
        return item


class LLMAnnotatorAgent:
    ALLOWED_TAGS = {
        "MODEL_RELEASE", "SDK_CHANGE", "OSS_RELEASE", "DEPRECATION",
        "CVE_SECURITY", "OUTAGE", "PRICING", "PRODUCT_FEATURE",
        "HIRING", "INFRA", "RESEARCH", "OTHER"
    }

    ALLOWED_TOPICS = {
        "models", "agents", "inference", "training", "gpu", "security", "policy",
        "open_source", "devtools", "products", "hiring"
    }

    def __init__(self, model: str):
        self.model = model or "qwen2.5:1.5b"

    def _extract_json(self, s: str):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _ollama_generate(self, prompt: str):
        url = OLLAMA_URL.rstrip("/") + "/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            r = requests.post(url, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "") or ""
        except Exception:
            return ""

    def annotate_item(self, item: dict):
        title = (item.get("title") or "").strip()
        source = (item.get("source") or "").strip()

        rule_tag = item.get("tag", "OTHER")
        rule_topics = item.get("topics", [])

        prompt = f"""
You are an assistant that classifies tech headlines into engineering-relevant AI signals.

Return ONLY valid JSON (no markdown, no extra text) with exactly these keys:
summary: string (1-2 sentences, neutral)
tag: one of [MODEL_RELEASE, SDK_CHANGE, OSS_RELEASE, DEPRECATION, CVE_SECURITY, OUTAGE, PRICING, PRODUCT_FEATURE, HIRING, INFRA, RESEARCH, OTHER]
topics: array of 0-4 from [models, agents, inference, training, gpu, security, policy, open_source, devtools, products, hiring]
action: string (one concrete next step an engineer can take)
confidence: integer 0-100
why: short string (why you picked this tag)

Rules:
- If this is NOT actually an AI/engineering signal, tag=OTHER, topics=[], confidence<=40.
- "exploitation" in a policy/abuse context is NOT CVE_SECURITY. CVE_SECURITY is for vulnerabilities/patching.
- Be conservative. If unsure, lower confidence.

Headline: {title}
Source: {source}
""".strip()

        raw = self._ollama_generate(prompt)
        obj = self._extract_json(raw)
        if not isinstance(obj, dict):
            item["llm_trace"] = {"agent": "LLMAnnotatorAgent", "ok": False, "model": self.model}
            return item

        tag = obj.get("tag", "OTHER")
        if tag not in self.ALLOWED_TAGS:
            tag = "OTHER"

        topics = obj.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        topics = [t for t in topics if t in self.ALLOWED_TOPICS][:4]

        summary = obj.get("summary", "")
        action = obj.get("action", "")
        why = obj.get("why", "")

        conf = obj.get("confidence", 0)
        try:
            conf = int(conf)
        except Exception:
            conf = 0
        conf = max(0, min(conf, 100))

        item["llm_summary"] = summary
        item["llm_tag"] = tag
        item["llm_topics"] = topics
        item["llm_action"] = action
        item["llm_confidence"] = conf
        item["llm_why"] = why
        item["llm_trace"] = {"agent": "LLMAnnotatorAgent", "ok": True, "model": self.model}

        if conf >= 75:
            item["tag"] = tag
            item["topics"] = topics
            if action:
                item["action"] = action

        item["llm_delta"] = {"rule_tag": rule_tag, "rule_topics": rule_topics}
        return item

    def annotate_topn(self, items: list, n: int):
        if not items or n <= 0:
            return items

        top = items[:n]
        rest = items[n:]

        def worker(x):
            return self.annotate_item(x)

        annotated = []
        with ThreadPoolExecutor(max_workers=min(6, n)) as ex:
            futs = [ex.submit(worker, x) for x in top]
            for fut in as_completed(futs):
                annotated.append(fut.result())

        by_key = {(it.get("url") or it.get("title")): it for it in annotated}
        ordered_top = []
        for it in top:
            k = it.get("url") or it.get("title")
            ordered_top.append(by_key.get(k, it))

        return ordered_top + rest


def diversify_and_select(items, k):
    by_source = defaultdict(list)
    for it in items:
        by_source[it.get("source", "Unknown")].append(it)

    for s in by_source:
        by_source[s].sort(key=lambda x: x.get("score", 0), reverse=True)

    out = []
    source_counts = Counter()
    sources = list(by_source.keys())
    idx = 0
    while len(out) < k and sources:
        src = sources[idx % len(sources)]
        if source_counts[src] < MAX_PER_SOURCE and by_source[src]:
            out.append(by_source[src].pop(0))
            source_counts[src] += 1
        sources = [s for s in sources if by_source[s]]
        idx += 1
    return out


def build_sections(selected):
    builder_tags = {"CVE_SECURITY", "OUTAGE", "DEPRECATION", "SDK_CHANGE", "INFRA", "OSS_RELEASE", "PRICING"}
    product_tags = {"PRODUCT_FEATURE", "PRICING", "OUTAGE"}

    builder_radar = [s for s in selected if s.get("tag") in builder_tags]
    product_watch = [s for s in selected if s.get("tag") in product_tags]
    action_queue = sorted(selected, key=lambda x: (x.get("priority") != "HIGH", -x.get("score", 0)))
    return builder_radar, product_watch, action_queue


def save_history_snapshot(signals):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H00")
    path = f"{HISTORY_DIR}/{stamp}.json"
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


def load_history(n=HISTORY_DAYS):
    ensure_dirs()
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
    files.sort()
    hist = []
    for fname in files[-n:]:
        hist.append(load_json(os.path.join(HISTORY_DIR, fname), default=None))
    return [h for h in hist if h and "signals" in h]


def compute_topic_trends(history):
    if not history:
        return []

    today = history[-1]["signals"]
    yesterday = history[-2]["signals"] if len(history) >= 2 else []

    def topic_counts(signals):
        c = Counter()
        for s in signals:
            for t in s.get("topics", []):
                c[t] += 1
        return c

    c_today = topic_counts(today)
    c_yest = topic_counts(yesterday)

    c_7d = Counter()
    for snap in history:
        for s in snap.get("signals", []):
            for t in s.get("topics", []):
                c_7d[t] += 1

    out = []
    for topic, count7 in c_7d.most_common():
        out.append({
            "topic": topic,
            "count_7d": count7,
            "count_today": c_today.get(topic, 0),
            "delta_vs_yesterday": (c_today.get(topic, 0) - c_yest.get(topic, 0)) if len(history) >= 2 else None
        })
    return out


def compute_new_since_last(history, selected):
    if not history or len(history) < 2:
        return selected[:10]
    prev = history[-2]["signals"]
    prev_keys = set((x.get("url") or x.get("title") or "").lower() for x in prev)
    out = []
    for s in selected:
        k = (s.get("url") or s.get("title") or "").lower()
        if k and k not in prev_keys:
            out.append(s)
    return out[:20]


def build_brief(selected):
    lines = ["Good morning.\n", "Here are the AI signals that actually matter today:\n"]
    for i, s in enumerate(selected[:10], 1):
        lines.append(f"{i}. {s.get('title')}")
        lines.append(
            f"   Source: {s.get('source')} • Tag: {s.get('tag')} • Priority: {s.get('priority')} • Change: {s.get('change_type')}"
        )
        if s.get("llm_summary"):
            lines.append(f"   Summary: {s.get('llm_summary')}")
        lines.append(f"   Action: {s.get('action')}\n")
    return "\n".join(lines).strip()


def export_daily_digest(payload: dict):
    ensure_dirs()
    generated = payload.get("generated_at_utc") or datetime.now(timezone.utc).isoformat()

    lines = []
    lines.append("# AI Radar — Daily Engineering Digest")
    lines.append(f"📅 {generated}\n")

    lines.append("## Executive Brief")
    for i, s in enumerate((payload.get("signals") or [])[:5], 1):
        lines.append(f"{i}. **{s.get('title')}**")
        lines.append(f"   Source: {s.get('source')}")
        lines.append(f"   Tag: {s.get('tag')} | Priority: {s.get('priority')}")
        if s.get("llm_summary"):
            lines.append(f"   Summary: {s.get('llm_summary')}")
        lines.append(f"   Action: {s.get('action')}\n")

    lines.append("## 🚨 High-Priority Alerts")
    high = [s for s in (payload.get("signals") or []) if s.get("priority") == "HIGH"]
    if not high:
        lines.append("- None today.")
    else:
        for s in high[:12]:
            lines.append(f"- [{s.get('tag')}] {s.get('title')}")

    lines.append("\n## 🧱 Builder Radar")
    br = payload.get("builder_radar") or []
    if not br:
        lines.append("- None today.")
    else:
        for s in br[:8]:
            lines.append(f"- {s.get('title')}")

    lines.append("\n## 🧭 Action Queue")
    aq = payload.get("action_queue") or []
    if not aq:
        lines.append("- None today.")
    else:
        for s in aq[:8]:
            lines.append(f"- {s.get('action')}")

    lines.append("\n## 📊 Topic Trends (7d)")
    tt = payload.get("topic_trends") or []
    if not tt:
        lines.append("- Not enough history yet.")
    else:
        for t in tt[:8]:
            d = t.get("delta_vs_yesterday")
            delta = "—" if d is None else (f"+{d}" if d > 0 else str(d))
            lines.append(f"- {t.get('topic')}: {t.get('count_7d')} (Δ {delta})")

    lines.append("\n---\nGenerated by AI Radar (local-first hybrid agent system)")

    with open(DAILY_DIGEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_pipeline():
    ensure_dirs()

    all_news = []
    for name, url in RSS_FEEDS.items():
        try:
            all_news.extend(fetch_rss(name, url, PER_FEED_LIMIT))
        except Exception:
            pass

    try:
        all_news.extend(fetch_hn_top(HN_LIMIT))
    except Exception:
        pass

    all_news = dedupe_items(all_news)
    save_json(RAW_NEWS_PATH, {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "items": all_news})

    memory = load_json(SIGNAL_MEMORY_PATH, default={})

    relevance_agent = AIRelevanceAgent()
    entity_agent = EntityAgent()
    tag_agent = TaggingAgent()
    topic_agent = TopicAgent()
    priority_agent = PriorityAgent()
    temporal_agent = TemporalChangeAgent(memory)
    action_agent = ActionAgent()

    kept = []
    dropped = []

    for it in all_news:
        ok, it = relevance_agent.process(it)
        if not ok:
            if ENABLE_DROPPED:
                it["drop_trace"] = {"agent": "AIRelevanceAgent", "reason": "Not AI-related"}
                dropped.append(it)
            continue

        it = entity_agent.process(it)
        it = tag_agent.process(it)
        it = topic_agent.process(it)
        it = priority_agent.process(it)
        it = temporal_agent.process(it)
        it = action_agent.process(it)
        kept.append(it)

    kept.sort(key=lambda x: x.get("score", 0), reverse=True)
    selected = diversify_and_select(kept, k=TOP_K_OVERALL)

    if ENABLE_LLM_ANNOTATOR:
        llm_agent = LLMAnnotatorAgent(LLM_MODEL)
        selected = llm_agent.annotate_topn(selected, LLM_ANNOTATE_TOP_N)

        for i in range(min(LLM_ANNOTATE_TOP_N, len(selected))):
            s = selected[i]
            if s.get("llm_confidence", 0) >= 75:
                s = priority_agent.process(s)

                if not (s.get("action") or "").strip():
                    s = action_agent.process(s)

                s["llm_override_applied"] = True
            else:
                s["llm_override_applied"] = False

            selected[i] = s

    builder_radar, product_watch, action_queue = build_sections(selected)
    save_history_snapshot(selected)
    history = load_history(HISTORY_DAYS)
    topics = compute_topic_trends(history)
    new_items = compute_new_since_last(history, selected)

    brief = build_brief(selected)

    temporal_agent.update(selected)
    save_json(SIGNAL_MEMORY_PATH, memory)

    payload = {
        "brief": brief,
        "signals": selected,
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

    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    export_daily_digest(payload)

    save_latest_snapshot(payload)
    return payload
