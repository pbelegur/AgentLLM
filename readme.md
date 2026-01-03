# AI Radar (Daily AI Brief) Local-first Hybrid Agent Pipeline

A **local-first “engineering radar”** that ingests tech news (RSS + Hacker News), applies **fast deterministic rule-agents** (filter/tag/topic/priority/temporal/action), then optionally uses a **FREE local LLM (Ollama)** to enrich only the **Top-N** signals with a better **summary + action + corrected tag/topics** (with guardrails).

No paid APIs.  
Runs on your laptop.  
Built for recruiter demos: *guardrails + explainability + UI.*

---

## What this project does

### 1) Ingestion
- Pulls headlines from:
  - RSS feeds (TechCrunch, Verge, WIRED, Ars, Gizmodo, CNET, Engadget, Mashable)
  - Hacker News top stories

### 2) Rule-agent pipeline (fast + deterministic)
Each item is processed by agents:

- **AIRelevanceAgent** → keep only AI/engineering-related signals
- **EntityAgent** → extract company mentions
- **TaggingAgent** → classify into one of:
  - `MODEL_RELEASE, SDK_CHANGE, OSS_RELEASE, DEPRECATION, CVE_SECURITY, OUTAGE, PRICING, PRODUCT_FEATURE, HIRING, INFRA, RESEARCH, OTHER`
- **TopicAgent** → topics like `models, inference, gpu, security, open_source, devtools, ...`
- **PriorityAgent** → score + priority (`HIGH/MED/LOW`)
- **TemporalChangeAgent** → `NEW/ONGOING/ESCALATING` based on memory
- **ActionAgent** → suggests a concrete engineering action

### 3) Optional LLM enrichment (FREE, local Ollama)
- Runs **ONLY on Top-N** ranked signals (default: 5)
- Adds structured fields:
  - `llm_summary, llm_tag, llm_topics, llm_action, llm_confidence, llm_why`
- **Guardrail**: rule outputs are only overridden if `confidence >= 75`
- If overridden, we recompute priority/score to keep ranking consistent

### 4) Web UI (Flask)
- `/ui` shows:
  - Brief
  - New since last snapshot
  - Builder Radar
  - Action Queue
  - Topic Tracker
  - Product Watch
- Includes **filters**:
  - Search
  - Priority
  - Tag
  - Source  
  (and a “Filtered Results” style experience depending on your latest UI version)

### 5) Storage & history
- Writes snapshots + memory into `data/`
  - `data/latest.json` (latest results)
  - `data/raw_news.json` (raw ingestion)
  - `data/signal_memory.json` (temporal memory)
  - `data/history/*.json` (history snapshots for trends)


## Requirements

- Python 3.10+ recommended
- Windows (works great on VS Code + PowerShell)
- Optional: **Ollama** (free local LLM)


### Create + activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate

### Create + activate a virtual environment
Start the Flask with UI python app.py
OPEN http://127.0.0.1:5000/ui

