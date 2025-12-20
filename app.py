from flask import Flask, jsonify, request, Response
import time
import json
import main

app = Flask(__name__)

# -------------------------
# SIMPLE CACHE (fast UI)
# -------------------------
CACHE_TTL_SECONDS = 120  # 2 minutes
_last_run_ts = 0.0
_last_result = None


def _get_bool(name: str, default: bool) -> bool:
    v = request.args.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    v = request.args.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except:
        return default


def _run_cached(force: bool = False):
    global _last_run_ts, _last_result
    now = time.time()

    if (not force) and _last_result is not None and (now - _last_run_ts) < CACHE_TTL_SECONDS:
        return _last_result, True

    data = main.run_pipeline()
    _last_result = data
    _last_run_ts = now
    return data, False


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return """
    <h3>AI Daily Brief is running ✅</h3>
    <ul>
      <li><a href="/ui">Open UI</a></li>
      <li><a href="/run">Run (JSON)</a></li>
    </ul>
    """


@app.route("/run")
def run():
    """
    /run?force=1&include_dropped=1&dropped_limit=30
    """
    force = _get_bool("force", False)
    include_dropped = _get_bool("include_dropped", False)
    dropped_limit = _get_int("dropped_limit", 30)

    data, cached = _run_cached(force=force)

    # Make response lighter by default
    if not include_dropped:
        data = dict(data)
        data["dropped"] = []
    else:
        if "dropped" in data and isinstance(data["dropped"], list):
            data = dict(data)
            data["dropped"] = data["dropped"][:dropped_limit]

    data["_meta"] = {
        "cached": cached,
        "cache_ttl_seconds": CACHE_TTL_SECONDS
    }

    return jsonify(data)


@app.route("/ui")
def ui():
    # UI loads fast; it calls /run in JS
    return Response(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Daily Brief</title>
  <style>
    :root{
      --bg1:#0b1020;
      --bg2:#111a33;
      --card:#121a2f;
      --card2:#0f162a;
      --text:#e8ecff;
      --muted:#aab3d6;
      --accent:#7c5cff;
      --accent2:#22c55e;
      --warn:#f59e0b;
      --danger:#ef4444;
      --border:rgba(255,255,255,.08);
      --shadow: 0 16px 40px rgba(0,0,0,.45);
      --radius: 18px;
    }

    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color:var(--text);
      background: radial-gradient(1000px 500px at 20% 0%, rgba(124,92,255,.35), transparent 55%),
                  radial-gradient(900px 600px at 80% 10%, rgba(34,197,94,.22), transparent 50%),
                  linear-gradient(180deg,var(--bg1),var(--bg2));
      min-height:100vh;
    }

    .wrap{max-width:1100px;margin:0 auto;padding:24px}
    .top{
      display:flex;gap:12px;align-items:center;justify-content:space-between;
      padding:18px 18px;border:1px solid var(--border);border-radius:var(--radius);
      background: rgba(18,26,47,.65);
      backdrop-filter: blur(10px);
      box-shadow: var(--shadow);
    }
    .title{
      display:flex;flex-direction:column;gap:4px;
    }
    .title h1{margin:0;font-size:20px;letter-spacing:.3px}
    .title p{margin:0;color:var(--muted);font-size:13px}

    .btns{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    button{
      border:none;border-radius:12px;padding:10px 12px;cursor:pointer;
      color:var(--text);background: rgba(255,255,255,.08);
      border:1px solid var(--border);
      transition: transform .06s ease, background .15s ease;
      font-weight:600;font-size:13px;
    }
    button:hover{background: rgba(255,255,255,.12)}
    button:active{transform: scale(.98)}
    .primary{background: rgba(124,92,255,.25);border-color: rgba(124,92,255,.45)}
    .primary:hover{background: rgba(124,92,255,.33)}
    .ok{background: rgba(34,197,94,.18);border-color: rgba(34,197,94,.35)}
    .warn{background: rgba(245,158,11,.18);border-color: rgba(245,158,11,.35)}

    .grid{display:grid;grid-template-columns: 1fr; gap:14px; margin-top:18px}
    .card{
      border:1px solid var(--border);
      border-radius:var(--radius);
      background: rgba(18,26,47,.62);
      backdrop-filter: blur(10px);
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .cardHeader{
      padding:14px 16px;
      display:flex;gap:12px;align-items:flex-start;justify-content:space-between;
      background: rgba(15,22,42,.7);
      border-bottom:1px solid var(--border);
    }
    .cardHeader a{color:var(--text);text-decoration:none}
    .cardHeader a:hover{text-decoration:underline}
    .cardHeader .h{
      display:flex;flex-direction:column;gap:6px;min-width:0;
    }
    .cardHeader .h .t{
      font-size:15px;font-weight:700;line-height:1.3;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      max-width: 760px;
    }
    .metaRow{display:flex;gap:8px;flex-wrap:wrap}
    .pill{
      font-size:11px;font-weight:800;letter-spacing:.3px;
      padding:5px 8px;border-radius:999px;border:1px solid var(--border);
      color:var(--text);background: rgba(255,255,255,.06);
    }
    .pill.accent{border-color: rgba(124,92,255,.6); background: rgba(124,92,255,.18)}
    .pill.ok{border-color: rgba(34,197,94,.55); background: rgba(34,197,94,.14)}
    .pill.warn{border-color: rgba(245,158,11,.55); background: rgba(245,158,11,.14)}
    .pill.danger{border-color: rgba(239,68,68,.55); background: rgba(239,68,68,.14)}

    .body{padding:14px 16px}
    .cols{display:grid;grid-template-columns: 1.2fr .8fr; gap:14px}
    .box{
      border:1px solid var(--border);
      border-radius:16px;
      background: rgba(255,255,255,.04);
      padding:12px;
    }
    .box h3{margin:0 0 8px 0;font-size:13px;color:var(--muted);font-weight:800;letter-spacing:.3px}
    .kv{display:flex;flex-wrap:wrap;gap:8px}
    .tag{
      font-size:12px;padding:6px 8px;border-radius:12px;
      background: rgba(255,255,255,.06);border:1px solid var(--border);
      color:var(--text);
    }
    .mono{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size:12px;color:rgba(232,236,255,.92);line-height:1.5;
      white-space:pre-wrap;
    }
    details{
      border:1px solid var(--border);
      border-radius:14px;
      background: rgba(255,255,255,.03);
      padding:10px;
      margin-top:10px;
    }
    summary{cursor:pointer;color:var(--muted);font-weight:800}
    .row{display:flex;gap:12px;align-items:center;justify-content:space-between;margin-top:14px}
    .small{font-size:12px;color:var(--muted)}
    .split{display:grid;grid-template-columns: 1fr 1fr; gap:14px; margin-top:16px}
    .sectionTitle{margin:18px 0 8px;color:var(--muted);font-weight:900;letter-spacing:.4px;font-size:12px;text-transform:uppercase}
    .empty{padding:18px;color:var(--muted)}
    .spinner{display:inline-block;width:14px;height:14px;border-radius:999px;border:2px solid rgba(255,255,255,.2);border-top-color:var(--accent);animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}

    @media (max-width: 900px){
      .cols{grid-template-columns:1fr}
      .cardHeader .h .t{max-width: 520px}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="title">
        <h1>AI Daily Brief</h1>
        <p>Signals • Temporal change • Impact • Transparent traces</p>
      </div>
      <div class="btns">
        <button class="primary" onclick="loadRun(false)">Run (cached)</button>
        <button class="warn" onclick="loadRun(true)">Force refresh</button>
        <button class="ok" onclick="toggleDropped()">Toggle dropped</button>
      </div>
    </div>

    <div class="row">
      <div class="small" id="status"><span class="spinner"></span> Loading…</div>
      <div class="small" id="meta"></div>
    </div>

    <div class="sectionTitle">Signals</div>
    <div id="signals" class="grid"></div>

    <div class="sectionTitle">Dropped items</div>
    <div id="dropped" class="grid"></div>
  </div>

<script>
let SHOW_DROPPED = false;

function esc(s){
  return (s ?? "").toString()
    .replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;").replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function pill(text, cls=""){
  return `<span class="pill ${cls}">${esc(text)}</span>`;
}

function pickTypeClass(t){
  if(!t) return "";
  if(t.includes("FOUNDATION")) return "accent";
  if(t.includes("INFRA")) return "ok";
  if(t.includes("HIRING")) return "warn";
  if(t.includes("POLICY")) return "warn";
  return "";
}

function pickChangeClass(c){
  if(c==="NEW") return "ok";
  if(c==="ONGOING") return "accent";
  if(c==="ESCALATING") return "danger";
  return "";
}

function renderSignal(item){
  const impact = item.impact || {};
  const roles = (impact.affected_roles || []).map(r => `<span class="tag">${esc(r)}</span>`).join("");
  const skills = (impact.skills || []).map(s => `<span class="tag">${esc(s)}</span>`).join("");
  const why = impact.why || "—";
  const llm = item.llm_reasoning || "—";

  const traces = {
    decision_trace: item.decision_trace || {},
    classification_trace: item.classification_trace || {},
    temporal_trace: item.temporal_trace || {},
  };

  return `
    <div class="card">
      <div class="cardHeader">
        <div class="h">
          <div class="t" title="${esc(item.title)}">
            <a href="${esc(item.url)}" target="_blank" rel="noreferrer">${esc(item.title)}</a>
          </div>
          <div class="metaRow">
            ${pill(item.signal_type || "UNKNOWN", pickTypeClass(item.signal_type || ""))}
            ${pill(item.change_type || "—", pickChangeClass(item.change_type || ""))}
            ${pill(item.source || "—")}
            ${pill("Interview: " + (impact.interview_relevance || "LOW"), (impact.interview_relevance==="HIGH" ? "danger" : ""))}
          </div>
        </div>
      </div>

      <div class="body">
        <div class="cols">
          <div class="box">
            <h3>Why it matters</h3>
            <div class="mono">${esc(why)}</div>
            <details>
              <summary>LLM reasoning (optional)</summary>
              <div class="mono" style="margin-top:10px">${esc(llm)}</div>
            </details>
          </div>

          <div class="box">
            <h3>Roles</h3>
            <div class="kv">${roles || '<span class="small">N/A</span>'}</div>
            <div style="height:10px"></div>
            <h3>Skills</h3>
            <div class="kv">${skills || '<span class="small">N/A</span>'}</div>
          </div>
        </div>

        <details>
          <summary>Agent traces</summary>
          <div class="mono" style="margin-top:10px">${esc(JSON.stringify(traces, null, 2))}</div>
        </details>
      </div>
    </div>
  `;
}

function renderDropped(item){
  const dt = item.drop_trace || {};
  const title = item.title || "—";
  const reason = dt.reason || "—";
  const agent = dt.agent || "—";

  return `
    <div class="card">
      <div class="cardHeader">
        <div class="h">
          <div class="t" title="${esc(title)}">${esc(title)}</div>
          <div class="metaRow">
            ${pill("Dropped", "danger")}
            ${pill(agent, "warn")}
            ${pill(item.source || "—")}
          </div>
        </div>
      </div>
      <div class="body">
        <div class="box">
          <h3>Reason</h3>
          <div class="mono">${esc(reason)}</div>
          <details>
            <summary>Details</summary>
            <div class="mono" style="margin-top:10px">${esc(JSON.stringify(dt.details || {}, null, 2))}</div>
          </details>
        </div>
      </div>
    </div>
  `;
}

async function loadRun(force){
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const signalsEl = document.getElementById("signals");
  const droppedEl = document.getElementById("dropped");

  statusEl.innerHTML = `<span class="spinner"></span> Running…`;

  const includeDropped = SHOW_DROPPED ? 1 : 0;
  const url = `/run?force=${force ? 1 : 0}&include_dropped=${includeDropped}&dropped_limit=30`;

  const start = performance.now();
  const res = await fetch(url);
  const data = await res.json();
  const took = Math.round(performance.now() - start);

  // Signals
  const signals = data.signals || [];
  signalsEl.innerHTML = signals.length
    ? signals.map(renderSignal).join("")
    : `<div class="empty">No signals found.</div>`;

  // Dropped (optional)
  const dropped = data.dropped || [];
  droppedEl.innerHTML = SHOW_DROPPED
    ? (dropped.length ? dropped.map(renderDropped).join("") : `<div class="empty">No dropped items in view.</div>`)
    : `<div class="empty">Hidden (click “Toggle dropped”).</div>`;

  // Status
  statusEl.textContent = `Done • ${signals.length} signals • ${SHOW_DROPPED ? dropped.length + " dropped shown" : "dropped hidden"} • ${took}ms`;
  const meta = data._meta || {};
  metaEl.textContent = meta.cached ? `cached ✅ (ttl ${meta.cache_ttl_seconds}s)` : `fresh 🔥 (ttl ${meta.cache_ttl_seconds}s)`;
}

function toggleDropped(){
  SHOW_DROPPED = !SHOW_DROPPED;
  loadRun(false);
}

loadRun(false);
</script>
</body>
</html>
        """,
        mimetype="text/html"
    )


if __name__ == "__main__":
    # IMPORTANT: keep debug off (you already are)
    app.run(host="127.0.0.1", port=5000)
