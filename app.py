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

    # If scheduler is running, serve the latest snapshot (fast + always updating)
    if not force:
        latest = _read_latest()
        if latest is not None:
            data = latest
            cached = True  # served from latest.json
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
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>AI Radar</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#0b1020; color:#e7e7e7; margin:0; }
    .wrap { max-width: 980px; margin: 22px auto; padding: 0 16px; }
    .top { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; padding:14px 14px;
           background:#101735; border-radius: 12px; border:1px solid rgba(255,255,255,.08); }
    .title { font-size:18px; font-weight:850; }
    .sub { font-size:12px; color: rgba(255,255,255,.65); margin-top:4px; }
    .btns { display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
    button { padding:9px 10px; border-radius:10px; border:1px solid rgba(255,255,255,.14);
             background:#141a33; color:#fff; cursor:pointer; font-weight:650; }
    button.secondary { background:#102636; }
    button.good { background:#10311f; }
    .meta { margin:10px 2px; font-size:12px; color: rgba(255,255,255,.7); }
    details { margin-top:12px; background:#0f1530; border-radius:12px; border:1px solid rgba(255,255,255,.08); }
    summary { padding:12px 14px; cursor:pointer; font-weight:800; }
    .box { padding: 0 14px 14px 14px; }
    .item { padding:10px 0; border-top:1px solid rgba(255,255,255,.08); }
    .item:first-child { border-top:none; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:6px; color: rgba(255,255,255,.7); font-size:12px; }
    .badge { display:inline-block; padding:2px 8px; border:1px solid rgba(255,255,255,.14); border-radius:999px; }
    a { color:#7dd3fc; text-decoration:none; }
    pre { white-space: pre-wrap; word-wrap: break-word; background:#0a0f22; padding:12px; border-radius:10px; overflow:auto; }
    .small { font-size:13px; color: rgba(255,255,255,.88); margin-top:6px; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="title">AI Radar</div>
      <div class="sub">Simple view: brief + lists (Builder, Actions, Topics, Products)</div>
    </div>
    <div class="btns">
      <button onclick="load(false)">Run (cached)</button>
      <button class="secondary" onclick="load(true)">Force refresh</button>
      <button class="good" onclick="toggleDebug()">Debug</button>
      <button onclick="toggleDropped()">Dropped</button>
    </div>
  </div>

  <div id="meta" class="meta"></div>

  <details open>
    <summary>Brief</summary>
    <div class="box">
      <pre id="brief"></pre>
    </div>
  </details>

  <details open>
    <summary>New since last snapshot</summary>
    <div class="box" id="new"></div>
  </details>

  <details>
    <summary>Builder Radar</summary>
    <div class="box" id="builder"></div>
  </details>

  <details>
    <summary>Action Queue</summary>
    <div class="box" id="actions"></div>
  </details>

  <details>
    <summary>Topic Tracker</summary>
    <div class="box" id="topics"></div>
  </details>

  <details>
    <summary>Product Watch</summary>
    <div class="box" id="products"></div>
  </details>

  <details id="droppedWrap" style="display:none;">
    <summary>Dropped</summary>
    <div class="box" id="dropped"></div>
  </details>

  <details id="rawWrap" style="display:none;">
    <summary>Debug JSON</summary>
    <div class="box"><pre id="raw"></pre></div>
  </details>
</div>

<script>
let DEBUG = false;
let SHOW_DROPPED = false;

function esc(s){
  return (s ?? "").toString()
    .replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function fmtItem(s, showAction=true){
  const title = esc(s.title || "");
  const url = esc(s.url || "#");
  const src = esc(s.source || "");
  const tag = esc(s.tag || "");
  const pr  = esc(s.priority || "");
  const ch  = esc(s.change_type || "");
  const topics = (s.topics || []).slice(0,6).map(t => `<span class="badge">${esc(t)}</span>`).join(" ");
  const action = esc(s.action || "");
  const summary = esc(s.llm_summary || "");

  return `
    <div class="item">
      <div style="font-weight:850;">
        <a href="${url}" target="_blank" rel="noreferrer">${title}</a>
      </div>
      <div class="row">
        <span class="badge">${src}</span>
        <span class="badge">${tag}</span>
        <span class="badge">${pr}</span>
        <span class="badge">${ch}</span>
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
      <div style="font-weight:850;">${topic}</div>
      <div class="row">
        <span class="badge">7d: ${c7}</span>
        <span class="badge">today: ${ct}</span>
        <span class="badge">Δ: ${esc(delta)}</span>
      </div>
    </div>
  `;
}

function toggleDebug(){ DEBUG = !DEBUG; load(false); }
function toggleDropped(){ SHOW_DROPPED = !SHOW_DROPPED; load(false); }

async function load(force){
  const mode = DEBUG ? "debug" : "product";
  const url = `/run?force=${force?1:0}&include_dropped=${SHOW_DROPPED?1:0}&mode=${mode}`;
  const res = await fetch(url);
  const data = await res.json();

  const meta = data._meta || {};
  const stats = data.stats || {};
  const updated = data.generated_at_utc ? ` • updated: ${data.generated_at_utc}` : "";
  document.getElementById("meta").textContent =
    `cached: ${meta.cached} • mode: ${meta.mode} • ttl: ${meta.ttl}s • raw: ${stats.raw_total} • selected: ${stats.selected}` + updated;

  document.getElementById("brief").textContent = data.brief || "";

  const newItems = (data.new_since_yesterday || data.signals || []).slice(0,10);
  document.getElementById("new").innerHTML =
    newItems.length ? newItems.map(s => fmtItem(s, true)).join("") : `<div class="item">No new items yet.</div>`;

  document.getElementById("builder").innerHTML =
    (data.builder_radar || []).slice(0,20).map(s => fmtItem(s, true)).join("") || `<div class="item">No builder signals.</div>`;

  document.getElementById("actions").innerHTML =
    (data.action_queue || []).slice(0,25).map(s => fmtItem(s, true)).join("") || `<div class="item">No actions.</div>`;

  document.getElementById("topics").innerHTML =
    (data.topic_trends || []).slice(0,25).map(fmtTopic).join("") || `<div class="item">No topic history yet.</div>`;

  document.getElementById("products").innerHTML =
    (data.product_watch || []).slice(0,20).map(s => fmtItem(s, true)).join("") || `<div class="item">No product signals.</div>`;

  // Dropped
  const droppedWrap = document.getElementById("droppedWrap");
  if(SHOW_DROPPED){
    droppedWrap.style.display = "block";
    const dropped = data.dropped || [];
    document.getElementById("dropped").innerHTML =
      dropped.length ? dropped.map(d => {
        const t = esc(d.title || "");
        const src = esc(d.source || "");
        const agent = esc((d.drop_trace && d.drop_trace.agent) || "");
        const reason = esc((d.drop_trace && d.drop_trace.reason) || "");
        return `<div class="item">
                  <div style="font-weight:850;">${t}</div>
                  <div class="row">
                    <span class="badge">${src}</span>
                    <span class="badge">${agent}</span>
                    <span class="badge">${reason}</span>
                  </div>
                </div>`;
      }).join("") : `<div class="item">No dropped items.</div>`;
  } else {
    droppedWrap.style.display = "none";
    document.getElementById("dropped").innerHTML = "";
  }

  // Debug JSON
  const rawWrap = document.getElementById("rawWrap");
  if(DEBUG){
    rawWrap.style.display = "block";
    document.getElementById("raw").textContent = JSON.stringify(data, null, 2);
  } else {
    rawWrap.style.display = "none";
    document.getElementById("raw").textContent = "";
  }
}

load(false);
// Refresh UI every 60s (reads latest snapshot; doesn't hammer RSS)
setInterval(() => load(false), 60000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
