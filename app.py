from flask import Flask, jsonify, request
import time
import main
import os
import json

app = Flask(__name__)

CACHE_TTL_SECONDS = 120
_last_run_ts = 0.0
_last_result = None


def _read_latest():
    try:
        if os.path.exists("data/latest.json"):
            with open("data/latest.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _get_bool(name: str, default: bool) -> bool:
    v = request.args.get(name, None)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _strip_heavy_fields(payload: dict) -> dict:
    """
    Product mode: remove traces to keep UI light.
    """
    p = dict(payload)

    def clean_signal(s: dict) -> dict:
        s = dict(s)
        s.pop("decision_trace", None)
        s.pop("temporal_trace", None)
        s.pop("tag_trace", None)
        s.pop("drop_trace", None)

  
        s.pop("llm_trace", None)
        return s

    for key in ["signals", "builder_radar", "product_watch", "action_queue", "new_since_yesterday"]:
        if key in p and isinstance(p[key], list):
            p[key] = [clean_signal(x) for x in p[key]]


    if "dropped" in p and isinstance(p["dropped"], list):
        short = []
        for d in p["dropped"][:120]:
            dd = {"title": d.get("title"), "source": d.get("source"), "url": d.get("url")}
            tr = d.get("drop_trace") or {}
            dd["drop_trace"] = {"agent": tr.get("agent"), "reason": tr.get("reason")}
            short.append(dd)
        p["dropped"] = short

    return p


def _run_cached(force: bool):
    global _last_run_ts, _last_result
    now = time.time()

    if (not force) and _last_result is not None and (now - _last_run_ts) < CACHE_TTL_SECONDS:
        return _last_result, True

    res = main.run_pipeline()
    _last_result = res
    _last_run_ts = now
    return res, False


@app.route("/")
def index():
    return "Running. Visit /ui"


@app.route("/run")
def run():
    force = _get_bool("force", False)
    include_dropped = _get_bool("include_dropped", False)
    mode = request.args.get("mode", "product").lower()

    if not force:
        latest = _read_latest()
        if latest is not None:
            data = latest
            cached = True
        else:
            data, cached = _run_cached(force=False)
    else:
        data, cached = _run_cached(force=True)

    if not include_dropped:
        data = dict(data)
        data["dropped"] = []

    if mode != "debug":
        data = _strip_heavy_fields(data)

    data["_meta"] = {"cached": cached, "ttl": CACHE_TTL_SECONDS, "mode": mode}
    return jsonify(data)


@app.route("/ui")
def ui():
    return r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AI Radar</title>
  <style>
    :root{
      --bg0:
      --bg1:
      --card: rgba(255,255,255,.06);
      --card2: rgba(255,255,255,.04);
      --border: rgba(255,255,255,.10);
      --text: rgba(255,255,255,.92);
      --muted: rgba(255,255,255,.70);
      --muted2: rgba(255,255,255,.55);
      --shadow: 0 18px 60px rgba(0,0,0,.55);
      --radius: 16px;
      --radius2: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }

    * { box-sizing: border-box; }
    body{
      margin:0;
      color: var(--text);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background:
        radial-gradient(1200px 600px at 20% 0%, rgba(125,211,252,.18), transparent 60%),
        radial-gradient(1000px 500px at 80% 10%, rgba(167,139,250,.14), transparent 60%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      min-height: 100vh;
    }

    a { color: 
    a:hover { text-decoration: underline; }

    .topbar{
      position: sticky; top: 0; z-index: 20;
      padding: 14px 0 12px 0;
      backdrop-filter: blur(10px);
      background: linear-gradient(180deg, rgba(7,10,20,.85), rgba(7,10,20,.55));
      border-bottom: 1px solid rgba(255,255,255,.06);
    }
    .toprow{
      display:flex; gap:12px; align-items:center; justify-content:space-between;
      max-width: 1120px; margin: 0 auto; padding: 0 16px;
    }
    .brand{
      display:flex; gap:12px; align-items:center;
    }
    .logo{
      width: 40px; height: 40px; border-radius: 12px;
      background: radial-gradient(circle at 30% 30%, rgba(125,211,252,.35), rgba(167,139,250,.22));
      border: 1px solid rgba(255,255,255,.10);
      box-shadow: 0 8px 30px rgba(0,0,0,.45);
    }
    .title{ font-weight: 900; letter-spacing: .2px; font-size: 18px; }
    .sub{ color: var(--muted2); font-size: 12px; margin-top: 2px; }

    .actions{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
    .btn{
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.06);
      color: var(--text);
      cursor: pointer;
      font-weight: 750;
      transition: transform .06s ease, background .15s ease, border .15s ease;
      user-select: none;
    }
    .btn:hover{ background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.20); }
    .btn:active{ transform: translateY(1px); }
    .btn.primary{ background: rgba(125,211,252,.12); border-color: rgba(125,211,252,.25); }
    .btn.good{ background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.25); }
    .btn.warn{ background: rgba(251,191,36,.10); border-color: rgba(251,191,36,.22); }

    .wrap { max-width: 1120px; margin: 0 auto; padding: 18px 16px 40px 16px; }

    .grid{
      display:grid;
      grid-template-columns: 360px 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    @media (max-width: 960px){
      .grid{ grid-template-columns: 1fr; }
    }

    .card{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card .hd{
      padding: 12px 14px;
      display:flex; align-items:center; justify-content:space-between; gap: 10px;
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
      border-bottom: 1px solid rgba(255,255,255,.06);
      font-weight: 900;
    }
    .card .bd{ padding: 12px 14px; }

    .meta{
      font-size: 12px; color: var(--muted);
      display:flex; gap:10px; flex-wrap:wrap; align-items:center;
      margin-top: 8px;
    }
    .pill{
      display:inline-flex; align-items:center; gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.04);
      font-size: 12px;
      color: rgba(255,255,255,.82);
    }
    .pill b{ color: var(--text); }
    .muted{ color: var(--muted2); }

    .field{
      display:flex; flex-direction:column; gap: 6px; margin-bottom: 10px;
    }
    .label{ font-size: 12px; color: var(--muted2); font-weight: 750; }
    input, select{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(10,15,34,.65);
      color: var(--text);
      outline: none;
    }
    input::placeholder{ color: rgba(255,255,255,.40); }
    input:focus, select:focus{ border-color: rgba(125,211,252,.35); box-shadow: 0 0 0 3px rgba(125,211,252,.10); }

    details{
      background: var(--card2);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: var(--radius2);
      overflow: hidden;
      margin-bottom: 10px;
    }
    summary{
      cursor:pointer;
      padding: 12px 14px;
      font-weight: 900;
      list-style: none;
      display:flex; justify-content:space-between; align-items:center;
      gap: 10px;
    }
    summary::-webkit-details-marker{ display:none; }
    .section{ padding: 0 14px 14px 14px; }

    .item{
      padding: 12px 0;
      border-top: 1px solid rgba(255,255,255,.08);
    }
    .item:first-child{ border-top:none; padding-top: 0; }
    .it-title{ font-weight: 900; line-height: 1.25; }
    .row{
      display:flex; gap: 8px; flex-wrap:wrap; align-items:center;
      margin-top: 8px;
      color: rgba(255,255,255,.72);
      font-size: 12px;
    }
    .badge{
      display:inline-flex; align-items:center;
      padding: 3px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.04);
    }

    .badge[data-pr="HIGH"]{ border-color: rgba(244,63,94,.30); background: rgba(244,63,94,.10); }
    .badge[data-pr="MED"] { border-color: rgba(251,191,36,.28); background: rgba(251,191,36,.09); }
    .badge[data-pr="LOW"] { border-color: rgba(148,163,184,.22); background: rgba(148,163,184,.06); }

    .badge[data-tag="CVE_SECURITY"], .badge[data-tag="OUTAGE"]{ border-color: rgba(244,63,94,.30); background: rgba(244,63,94,.10); }
    .badge[data-tag="DEPRECATION"], .badge[data-tag="SDK_CHANGE"], .badge[data-tag="PRICING"]{ border-color: rgba(251,191,36,.28); background: rgba(251,191,36,.09); }
    .badge[data-tag="MODEL_RELEASE"], .badge[data-tag="INFRA"]{ border-color: rgba(125,211,252,.28); background: rgba(125,211,252,.09); }
    .badge[data-tag="RESEARCH"], .badge[data-tag="OSS_RELEASE"]{ border-color: rgba(167,139,250,.28); background: rgba(167,139,250,.09); }

    .small{
      font-size: 13px;
      color: rgba(255,255,255,.88);
      margin-top: 8px;
      line-height: 1.35;
    }
    .small b{ color: var(--text); }

    pre{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: rgba(10,15,34,.65);
      border: 1px solid rgba(255,255,255,.10);
      padding: 12px;
      border-radius: 12px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(255,255,255,.88);
    }

    .footer-note{
      margin-top: 12px;
      font-size: 12px;
      color: rgba(255,255,255,.55);
      text-align: center;
    }
  </style>
</head>

<body>
  <div class="topbar">
    <div class="toprow">
      <div class="brand">
        <div class="logo"></div>
        <div>
          <div class="title">AI Radar</div>
          <div class="sub">Local-first hybrid agent pipeline (rules + optional Ollama annotator)</div>
        </div>
      </div>

      <div class="actions">
        <button class="btn" onclick="load(false)">Run (cached)</button>
        <button class="btn primary" onclick="load(true)">Force refresh</button>
        <button class="btn good" onclick="toggleDebug()">Debug</button>
        <button class="btn" onclick="toggleDropped()">Dropped</button>
      </div>
    </div>
  </div>

  <div class="wrap">
    <div class="grid">
      <!-- LEFT: Filters -->
      <div class="card">
        <div class="hd">
          <span>Filters</span>
          <button class="btn warn" style="padding:8px 10px;" onclick="clearFilters()">Clear</button>
        </div>
        <div class="bd">
          <div class="field">
            <div class="label">Search</div>
            <input id="q" placeholder="e.g., OpenAI, NVIDIA, inference, pricing..." oninput="render()"/>
          </div>

          <div class="field">
            <div class="label">Priority</div>
            <select id="priority" onchange="render()">
              <option value="">All</option>
              <option value="HIGH">HIGH</option>
              <option value="MED">MED</option>
              <option value="LOW">LOW</option>
            </select>
          </div>

          <div class="field">
            <div class="label">Tag</div>
            <select id="tag" onchange="render()">
              <option value="">All</option>
            </select>
          </div>

          <div class="field">
            <div class="label">Source</div>
            <select id="source" onchange="render()">
              <option value="">All</option>
            </select>
          </div>

          <div class="meta" id="meta"></div>
          <div class="footer-note" style="margin-top:10px;">
            Tip: Brief is static. Use <b>Filtered Results</b> to see search working.
          </div>
        </div>
      </div>

      <!-- RIGHT: Content -->
      <div class="card">
        <div class="hd">
          <span>Today</span>
          <span class="pill"><span class="muted">view:</span> <b id="viewMode">product</b></span>
        </div>
        <div class="bd">
          <details open>
            <summary>Brief <span class="pill" id="briefCount"></span></summary>
            <div class="section">
              <pre id="brief"></pre>
            </div>
          </details>

          <!-- NEW: This is what makes search/filter feel real -->
          <details open>
            <summary>Filtered Results <span class="pill" id="resultsCount"></span></summary>
            <div class="section" id="results"></div>
          </details>

          <details open>
            <summary>New since last snapshot <span class="pill" id="newCount"></span></summary>
            <div class="section" id="new"></div>
          </details>

          <details>
            <summary>Builder Radar <span class="pill" id="builderCount"></span></summary>
            <div class="section" id="builder"></div>
          </details>

          <details>
            <summary>Action Queue <span class="pill" id="actionsCount"></span></summary>
            <div class="section" id="actions"></div>
          </details>

          <details>
            <summary>Topic Tracker <span class="pill" id="topicsCount"></span></summary>
            <div class="section" id="topics"></div>
          </details>

          <details>
            <summary>Product Watch <span class="pill" id="productsCount"></span></summary>
            <div class="section" id="products"></div>
          </details>

          <details id="droppedWrap" style="display:none;">
            <summary>Dropped <span class="pill" id="droppedCount"></span></summary>
            <div class="section" id="dropped"></div>
          </details>

          <details id="rawWrap" style="display:none;">
            <summary>Debug JSON</summary>
            <div class="section"><pre id="raw"></pre></div>
          </details>

          <div class="footer-note">Auto-refresh every 60s (reads latest snapshot; doesn’t hammer RSS).</div>
        </div>
      </div>
    </div>
  </div>

<script>
let DEBUG = false;
let SHOW_DROPPED = false;
let DATA = null;

function esc(s){
  return (s ?? "").toString()
    .replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function badge(txt, attrs={}){
  const a = Object.entries(attrs).map(([k,v]) => ` data-${k}="${esc(v)}"`).join("");
  return `<span class="badge"${a}>${esc(txt)}</span>`;
}

function fmtItem(s, showAction=true){
  const title = esc(s.title || "");
  const url = esc(s.url || "#");
  const src = s.source || "";
  const tag = s.tag || "";
  const pr  = s.priority || "";
  const ch  = s.change_type || "";
  const topics = (s.topics || []).slice(0,6).map(t => badge(t)).join(" ");
  const action = esc(s.action || "");
  const summary = esc(s.llm_summary || "");

  return `
    <div class="item">
      <div class="it-title">
        <a href="${url}" target="_blank" rel="noreferrer">${title}</a>
      </div>
      <div class="row">
        ${badge(src)}
        ${badge(tag, {tag: tag})}
        ${badge(pr, {pr: pr})}
        ${badge(ch)}
      </div>
      ${topics ? `<div class="row">${topics}</div>` : ""}
      ${summary ? `<div class="small"><b>Summary:</b> ${summary}</div>` : ""}
      ${showAction && action ? `<div class="small"><b>Action:</b> ${action}</div>` : ""}
    </div>
  `;
}

function fmtTopic(t){
  const topic = esc(t.topic);
  const c7 = esc(t.count_7d);
  const ct = esc(t.count_today);
  const d = t.delta_vs_yesterday;
  const delta = (d === null || d === undefined) ? "—" : (d > 0 ? `+${d}` : `${d}`);
  return `
    <div class="item">
      <div class="it-title">${topic}</div>
      <div class="row">
        ${badge(`7d: ${c7}`)}
        ${badge(`today: ${ct}`)}
        ${badge(`Δ: ${delta}`)}
      </div>
    </div>
  `;
}

function toggleDebug(){ DEBUG = !DEBUG; load(false); }
function toggleDropped(){ SHOW_DROPPED = !SHOW_DROPPED; load(false); }

function clearFilters(){
  document.getElementById("q").value = "";
  document.getElementById("priority").value = "";
  document.getElementById("tag").value = "";
  document.getElementById("source").value = "";
  render();
}

function applyFilters(list){
  const q = (document.getElementById("q").value || "").trim().toLowerCase();
  const pr = document.getElementById("priority").value;
  const tag = document.getElementById("tag").value;
  const src = document.getElementById("source").value;

  return (list || []).filter(s => {
    if (pr && (s.priority || "") !== pr) return false;
    if (tag && (s.tag || "") !== tag) return false;
    if (src && (s.source || "") !== src) return false;
    if (q){
      const hay = `${s.title||""} ${(s.llm_summary||"")} ${(s.action||"")} ${(s.source||"")} ${(s.tag||"")}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function fillSelectOptions(id, values){
  const sel = document.getElementById(id);
  const cur = sel.value || "";
  const opts = ["", ...Array.from(new Set(values.filter(Boolean))).sort()];
  sel.innerHTML = opts.map(v => `<option value="${esc(v)}">${v ? esc(v) : "All"}</option>`).join("");
  if (opts.includes(cur)) sel.value = cur;
}

function setCount(id, n){
  document.getElementById(id).innerHTML = `<span class="muted">count:</span> <b>${n}</b>`;
}

function render(){
  if(!DATA) return;

  document.getElementById("viewMode").textContent = DEBUG ? "debug" : "product";

  const signals = applyFilters(DATA.signals || []);
  const builder = applyFilters(DATA.builder_radar || []);
  const actions = applyFilters(DATA.action_queue || []);
  const products = applyFilters(DATA.product_watch || []);
  const newItems = applyFilters((DATA.new_since_yesterday || DATA.signals || []).slice(0,10));

  document.getElementById("brief").textContent = DATA.brief || "";

  // NEW: show filtered list so search is visibly working
  document.getElementById("results").innerHTML =
    signals.length
      ? signals.slice(0,30).map(s => fmtItem(s, true)).join("")
      : `<div class="item">No matches. Try clearing Priority/Tag/Source.</div>`;

  document.getElementById("new").innerHTML =
    newItems.length ? newItems.map(s => fmtItem(s, true)).join("") : `<div class="item">No new items yet.</div>`;

  document.getElementById("builder").innerHTML =
    builder.slice(0,20).map(s => fmtItem(s, true)).join("") || `<div class="item">No builder signals.</div>`;

  document.getElementById("actions").innerHTML =
    actions.slice(0,25).map(s => fmtItem(s, true)).join("") || `<div class="item">No actions.</div>`;

  document.getElementById("topics").innerHTML =
    (DATA.topic_trends || []).slice(0,25).map(fmtTopic).join("") || `<div class="item">No topic history yet.</div>`;

  document.getElementById("products").innerHTML =
    products.slice(0,20).map(s => fmtItem(s, true)).join("") || `<div class="item">No product signals.</div>`;

  const droppedWrap = document.getElementById("droppedWrap");
  if(SHOW_DROPPED){
    droppedWrap.style.display = "block";
    const dropped = DATA.dropped || [];
    document.getElementById("dropped").innerHTML =
      dropped.length ? dropped.slice(0,120).map(d => {
        const t = esc(d.title || "");
        const src = esc(d.source || "");
        const agent = esc((d.drop_trace && d.drop_trace.agent) || "");
        const reason = esc((d.drop_trace && d.drop_trace.reason) || "");
        return `<div class="item">
                  <div class="it-title">${t}</div>
                  <div class="row">
                    ${badge(src)}
                    ${badge(agent)}
                    ${badge(reason)}
                  </div>
                </div>`;
      }).join("") : `<div class="item">No dropped items.</div>`;
  } else {
    droppedWrap.style.display = "none";
    document.getElementById("dropped").innerHTML = "";
  }

  const rawWrap = document.getElementById("rawWrap");
  if(DEBUG){
    rawWrap.style.display = "block";
    document.getElementById("raw").textContent = JSON.stringify(DATA, null, 2);
  } else {
    rawWrap.style.display = "none";
    document.getElementById("raw").textContent = "";
  }

  setCount("briefCount", (DATA.signals || []).length);
  setCount("resultsCount", signals.length);
  setCount("newCount", newItems.length);
  setCount("builderCount", builder.length);
  setCount("actionsCount", actions.length);
  setCount("topicsCount", (DATA.topic_trends || []).length);
  setCount("productsCount", products.length);
  setCount("droppedCount", (DATA.dropped || []).length);

  const meta = DATA._meta || {};
  const stats = DATA.stats || {};
  const updated = DATA.generated_at_utc ? `updated: ${DATA.generated_at_utc}` : "";
  document.getElementById("meta").innerHTML = `
    <span class="pill"><span class="muted">cached:</span> <b>${meta.cached}</b></span>
    <span class="pill"><span class="muted">ttl:</span> <b>${meta.ttl}s</b></span>
    <span class="pill"><span class="muted">raw:</span> <b>${stats.raw_total ?? "-"}</b></span>
    <span class="pill"><span class="muted">selected:</span> <b>${stats.selected ?? "-"}</b></span>
    <span class="pill"><b>${esc(updated)}</b></span>
  `;
}

async function load(force){
  const mode = DEBUG ? "debug" : "product";
  const url = `/run?force=${force?1:0}&include_dropped=${SHOW_DROPPED?1:0}&mode=${mode}`;
  const res = await fetch(url);
  const data = await res.json();
  DATA = data;

  const tags = (data.signals || []).map(s => s.tag);
  const sources = (data.signals || []).map(s => s.source);
  fillSelectOptions("tag", tags);
  fillSelectOptions("source", sources);

  render();
}

load(false);
setInterval(() => load(false), 60000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
