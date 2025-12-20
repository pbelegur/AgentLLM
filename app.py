from flask import Flask, jsonify, request
import time
import main

app = Flask(__name__)

CACHE_TTL_SECONDS = 120
_last_run_ts = 0.0
_last_result = None


def _get_bool(name: str, default: bool) -> bool:
    v = request.args.get(name, None)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    v = request.args.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


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
        return s

    for key in ["signals", "builder_radar", "product_watch", "action_queue", "new_since_yesterday"]:
        if key in p and isinstance(p[key], list):
            p[key] = [clean_signal(x) for x in p[key]]

    # dropped: keep only agent+reason
    if "dropped" in p and isinstance(p["dropped"], list):
        short = []
        for d in p["dropped"][:80]:
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

    data, cached = _run_cached(force=force)

    if not include_dropped:
        data = dict(data)
        data["dropped"] = []

    if mode != "debug":
        data = _strip_heavy_fields(data)

    data["_meta"] = {"cached": cached, "ttl": CACHE_TTL_SECONDS, "mode": mode}
    return jsonify(data)


@app.route("/ui")
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>AI Radar</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b1020; color:#e7e7e7; margin:0; }
    .wrap { max-width: 1200px; margin: 26px auto; padding: 0 18px; }
    .bar { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:16px 18px;
           background: linear-gradient(90deg, #1a1f3a, #0b2a2a); border-radius: 14px; border:1px solid rgba(255,255,255,.08);}
    .btn { padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.14);
           background:#141a33; color:#fff; cursor:pointer; font-weight:700; }
    .btn.secondary { background:#102636; }
    .btn.good { background:#10311f; }
    .tabs { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }
    .tab { padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.14);
           background:#0f1530; cursor:pointer; font-weight:700; }
    .tab.active { background:#18224a; }
    .muted { color: rgba(255,255,255,.7); font-size: 13px; }
    .card { margin-top:14px; padding:16px; border-radius:14px; background:#0f1530; border:1px solid rgba(255,255,255,.08); }
    .pill { display:inline-block; padding:3px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.12);
            font-size:12px; margin-right:8px; color: rgba(255,255,255,.85); }
    a { color:#7dd3fc; text-decoration:none; }
    pre { white-space: pre-wrap; word-wrap: break-word; background:#0a0f22; padding:12px; border-radius:12px; overflow:auto; }
    .grid { display:grid; grid-template-columns: 1fr; gap: 12px; }
    @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div>
        <div style="font-size:20px;font-weight:900;">AI Radar</div>
        <div class="muted">Builder • Actions • Topics • Products</div>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <button class="btn" onclick="load(false)">Run (cached)</button>
        <button class="btn secondary" onclick="load(true)">Force refresh</button>
        <button class="btn good" onclick="toggleDebug()">Toggle debug</button>
        <button class="btn" onclick="toggleDropped()">Toggle dropped</button>
      </div>
    </div>

    <div id="meta" class="muted" style="margin-top:10px;"></div>

    <div class="tabs">
      <div class="tab active" id="tab-summary" onclick="setTab('summary')">Summary</div>
      <div class="tab" id="tab-builder" onclick="setTab('builder')">Builder Radar</div>
      <div class="tab" id="tab-actions" onclick="setTab('actions')">Action Queue</div>
      <div class="tab" id="tab-topics" onclick="setTab('topics')">Topic Tracker</div>
      <div class="tab" id="tab-products" onclick="setTab('products')">Product Watch</div>
    </div>

    <div id="content"></div>
    <div id="raw" class="card" style="display:none;"></div>
  </div>

<script>
let DEBUG = false;
let SHOW_DROPPED = false;
let TAB = "summary";

function esc(s){
  return (s ?? "").toString()
    .replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;").replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function setTab(t){
  TAB = t;
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  document.getElementById("tab-"+t).classList.add("active");
  load(false);
}

function toggleDebug(){ DEBUG = !DEBUG; load(false); }
function toggleDropped(){ SHOW_DROPPED = !SHOW_DROPPED; load(false); }

function renderSignalCard(s){
  const title = esc(s.title);
  const url = esc(s.url || "#");
  const src = esc(s.source || "");
  const tag = esc(s.tag || "");
  const pr = esc(s.priority || "");
  const ch = esc(s.change_type || "");
  const topics = (s.topics || []).map(t => `<span class="pill">${esc(t)}</span>`).join("");
  const action = esc(s.action || "");

  return `
    <div class="card">
      <div style="font-size:16px;font-weight:900;">
        <a href="${url}" target="_blank" rel="noreferrer">${title}</a>
      </div>
      <div class="muted" style="margin-top:6px;">
        <span class="pill">${src}</span>
        <span class="pill">${tag}</span>
        <span class="pill">${pr}</span>
        <span class="pill">${ch}</span>
      </div>
      <div style="margin-top:8px;">${topics}</div>
      <div style="margin-top:10px;"><b>Action:</b> ${action}</div>
    </div>
  `;
}

function renderDropped(dropped){
  if(!SHOW_DROPPED) return "";
  const items = (dropped || []).map(d => {
    const t = esc(d.title || "");
    const src = esc(d.source || "");
    const agent = esc((d.drop_trace && d.drop_trace.agent) || "");
    const reason = esc((d.drop_trace && d.drop_trace.reason) || "");
    return `<div style="margin-bottom:10px;">
              <div style="font-weight:800;">${t}</div>
              <div class="muted">${src} • ${agent} • ${reason}</div>
            </div>`;
  }).join("");
  return `<div class="card"><div style="font-weight:900;margin-bottom:8px;">Dropped</div>${items}</div>`;
}

function renderTopics(trends){
  const rows = (trends || []).map(t => {
    const topic = esc(t.topic);
    const c7 = esc(t.count_7d);
    const ct = esc(t.count_today);
    const d = t.delta_vs_yesterday;
    const delta = d > 0 ? `+${d}` : `${d}`;
    return `<div class="card">
              <div style="font-weight:900;">${topic}</div>
              <div class="muted">7d: ${c7} • today: ${ct} • Δ vs yesterday: ${esc(delta)}</div>
            </div>`;
  }).join("");
  return `<div class="grid">${rows}</div>`;
}

async function load(force){
  const mode = DEBUG ? "debug" : "product";
  const url = `/run?force=${force?1:0}&include_dropped=${SHOW_DROPPED?1:0}&mode=${mode}`;
  const res = await fetch(url);
  const data = await res.json();

  const meta = data._meta || {};
  const stats = data.stats || {};
  document.getElementById("meta").textContent =
    `cached: ${meta.cached} • mode: ${meta.mode} • ttl: ${meta.ttl}s • raw: ${stats.raw_total} • selected: ${stats.selected}`;

  let html = "";

  if(TAB === "summary"){
    html += `<div class="card"><div style="font-weight:900;margin-bottom:8px;">Brief</div><pre>${esc(data.brief)}</pre></div>`;
    html += `<div class="card"><div style="font-weight:900;margin-bottom:8px;">New since yesterday</div>` +
            (data.new_since_yesterday || []).slice(0,10).map(renderSignalCard).join("") +
            `</div>`;
    html += renderDropped(data.dropped);

  } else if(TAB === "builder"){
    html += `<div class="card"><div style="font-weight:900;margin-bottom:8px;">Builder Radar</div></div>`;
    html += (data.builder_radar || []).map(renderSignalCard).join("");
    html += renderDropped(data.dropped);

  } else if(TAB === "actions"){
    html += `<div class="card"><div style="font-weight:900;margin-bottom:8px;">Action Queue</div><div class="muted">Sorted by priority + score</div></div>`;
    html += (data.action_queue || []).map(renderSignalCard).join("");
    html += renderDropped(data.dropped);

  } else if(TAB === "topics"){
    html += `<div class="card"><div style="font-weight:900;margin-bottom:8px;">Topic Tracker</div><div class="muted">Counts from saved daily history</div></div>`;
    html += renderTopics(data.topic_trends || []);
    html += renderDropped(data.dropped);

  } else if(TAB === "products"){
    html += `<div class="card"><div style="font-weight:900;margin-bottom:8px;">Product Watch</div></div>`;
    html += (data.product_watch || []).map(renderSignalCard).join("");
    html += renderDropped(data.dropped);
  }

  document.getElementById("content").innerHTML = html;

  const raw = document.getElementById("raw");
  if(DEBUG){
    raw.style.display = "block";
    raw.innerHTML = `<div style="font-weight:900;margin-bottom:8px;">Debug JSON</div><pre>${esc(JSON.stringify(data, null, 2))}</pre>`;
  } else {
    raw.style.display = "none";
    raw.innerHTML = "";
  }
}

load(false);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
