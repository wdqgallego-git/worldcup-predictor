"""Build a self-contained HTML dashboard from the current pick sheet and live results.

Writes outputs/penka_dashboard.html: no network, no dependencies, openable in any
browser (or shareable into a Claude chat). Re-run after every scoreboard update.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "penka_dashboard.html"


def build_rows() -> tuple[list[dict], str]:
    sheet = pd.read_csv(PROJECT_ROOT / "outputs" / "pick_sheet.csv")
    rec = pd.read_csv(PROJECT_ROOT / "outputs" / "final_match_recommendations.csv")
    live_path = PROJECT_ROOT / "data" / "raw" / "live_results_2026.csv"
    live = pd.read_csv(live_path, dtype=str) if live_path.exists() else pd.DataFrame(columns=["match_id", "goals_a", "goals_b"])
    live["match_id"] = live["match_id"].astype(int)
    merged = sheet.merge(
        rec[["match_id", "safe_prediction", "rank_a", "rank_b", "recommended_strategy"]], on="match_id"
    ).merge(live[["match_id", "goals_a", "goals_b"]], on="match_id", how="left")
    rows = []
    for r in merged.itertuples(index=False):
        played = pd.notna(r.goals_a) and str(r.goals_a).strip() != ""
        rows.append(
            {
                "id": int(r.match_id),
                "g": str(r.group).replace("Group ", ""),
                "d": str(r.date),
                "t": str(r.time),
                "a": r.team_a,
                "b": r.team_b,
                "rec": r.recommended_prediction,
                "model": r.safe_prediction,
                "fresh": r.odds_freshness,
                "hrs": None if pd.isna(r.market_hours_before_kickoff) else round(float(r.market_hours_before_kickoff), 1),
                "pa": None if pd.isna(r.market_prob_team_a) else round(float(r.market_prob_team_a), 3),
                "pd": None if pd.isna(r.market_prob_draw) else round(float(r.market_prob_draw), 3),
                "pb": None if pd.isna(r.market_prob_team_b) else round(float(r.market_prob_team_b), 3),
                "score": f"{int(float(r.goals_a))}-{int(float(r.goals_b))}" if played else None,
            }
        )
    strategy = str(merged["recommended_strategy"].iloc[0])
    return rows, strategy


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Penka 2026 picks dashboard</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#6b6b6b;--bd:#e3e1da;--card:#f5f4ef;--grn-bg:#EAF3DE;--grn-fg:#27500A;--amb-bg:#FAEEDA;--amb-fg:#633806;--teal:#5DCAA5;--gray:#B4B2A9;--coral:#F0997B;}
@media (prefers-color-scheme: dark){:root{--bg:#1c1c1a;--fg:#ececea;--mut:#9b9b96;--bd:#3a3a36;--card:#262624;--grn-bg:#27500A;--grn-fg:#C0DD97;--amb-bg:#633806;--amb-fg:#FAC775;}}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;max-width:760px;margin-inline:auto;}
h1{font-size:20px;font-weight:600;margin:0 0 4px;}
.sub{font-size:12px;color:var(--mut);margin:0 0 16px;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:14px;}
.card{background:var(--card);border-radius:10px;padding:12px;}
.card .l{font-size:12px;color:var(--mut);}
.card .v{font-size:20px;font-weight:600;margin-top:2px;}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;}
button,select{font-size:13px;padding:6px 12px;border:1px solid var(--bd);border-radius:8px;background:transparent;color:var(--fg);cursor:pointer;}
button.on{background:var(--card);}
.row{display:grid;grid-template-columns:78px minmax(0,1.4fr) 62px 52px minmax(80px,1fr) auto;gap:8px;align-items:center;padding:10px 6px;border-bottom:1px solid var(--bd);}
.dt{font-size:12px;font-weight:600;}.tm{font-size:11px;color:var(--mut);}
.teams{font-size:13px;}
.pick .l{font-size:10px;color:var(--mut);text-align:center;}.pick .v{font-size:16px;font-weight:600;text-align:center;}
.model .v{font-size:13px;color:var(--mut);font-weight:400;}
.bar{display:flex;height:7px;border-radius:4px;overflow:hidden;}
.pct{font-size:10px;color:var(--mut);margin-top:2px;}
.chip{font-size:11px;padding:3px 9px;border-radius:999px;white-space:nowrap;}
.chip.g{background:var(--grn-bg);color:var(--grn-fg);}.chip.a{background:var(--amb-bg);color:var(--amb-fg);}.chip.n{background:var(--card);color:var(--mut);}
.scorechip{font-size:12px;font-weight:600;background:var(--card);padding:3px 9px;border-radius:999px;}
.hint{font-size:11px;color:var(--mut);margin-top:14px;line-height:1.6;}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:11px;}
</style></head><body>
<h1>Penka 2026 — group-stage picks</h1>
<p class="sub">strategy: __STRATEGY__ &middot; generated __GENERATED__ &middot; refresh with <code>python scripts/build_dashboard_html.py</code></p>
<div class="cards">
<div class="card"><div class="l">Fixtures</div><div class="v" id="m-total"></div></div>
<div class="card"><div class="l">Played</div><div class="v" id="m-played"></div></div>
<div class="card"><div class="l">Fresh odds</div><div class="v" id="m-fresh" style="color:var(--grn-fg)"></div></div>
<div class="card"><div class="l">Stale / no odds</div><div class="v" id="m-stale" style="color:var(--amb-fg)"></div></div>
</div>
<div class="filters">
<button id="f-next" class="on">Next 3 days</button>
<button id="f-all">All 72</button>
<button id="f-played">Played</button>
<select id="f-group"><option value="">All groups</option></select>
</div>
<div id="rows"></div>
<p class="hint">Submit = recommended pick. Model = EV optimizer pick for reference. Bars: market win / draw / win.
Stale matches: update the closing line in <code>data/raw/match_odds.csv</code> ~2h before kickoff, run
<code>python scripts/reoptimize_match.py --match N</code>, re-apply your strategy override, then rebuild this file.</p>
<script>
const M=__DATA__;
document.getElementById('m-total').textContent=M.length;
document.getElementById('m-played').textContent=M.filter(m=>m.score).length;
document.getElementById('m-fresh').textContent=M.filter(m=>m.fresh==='green').length;
document.getElementById('m-stale').textContent=M.filter(m=>m.fresh!=='green').length;
const sel=document.getElementById('f-group');
[...new Set(M.map(m=>m.g))].sort().forEach(g=>{const o=document.createElement('option');o.value=g;o.textContent='Group '+g;sel.appendChild(o);});
function chip(m){
  if(m.fresh==='none')return '<span class="chip n">no odds</span>';
  const txt=m.hrs<=6?Math.round(m.hrs)+'h pre-KO':(m.hrs<=48?'captured '+Math.round(m.hrs)+'h ago':'captured '+Math.round(m.hrs/24)+'d ago');
  return '<span class="chip '+(m.fresh==='green'?'g':'a')+'">'+txt+'</span>';
}
function bar(m){
  if(m.pa===null)return '<div class="pct">model-only</div>';
  const pa=Math.round(m.pa*100),pd=Math.round(m.pd*100),pb=100-pa-pd;
  return '<div class="bar"><div style="width:'+pa+'%;background:var(--teal)"></div><div style="width:'+pd+'%;background:var(--gray)"></div><div style="width:'+pb+'%;background:var(--coral)"></div></div><div class="pct">'+pa+' / '+pd+' / '+pb+'%</div>';
}
function row(m){
  const right=m.score?'<span class="scorechip">FT '+m.score+'</span>':chip(m);
  return '<div class="row"><div><div class="dt">'+m.d.slice(5)+'</div><div class="tm">'+m.t+' · Grp '+m.g+'</div></div>'
  +'<div class="teams">'+m.a+' <span style="color:var(--mut)">vs</span> '+m.b+'</div>'
  +'<div class="pick"><div class="l">submit</div><div class="v">'+m.rec+'</div></div>'
  +'<div class="pick model"><div class="l">model</div><div class="v">'+m.model+'</div></div>'
  +'<div>'+bar(m)+'</div><div>'+right+'</div></div>';
}
let mode='next';
function render(){
  let list=M.slice();
  const g=sel.value;if(g)list=list.filter(m=>m.g===g);
  if(mode==='next'){const days=[...new Set(M.filter(m=>!m.score).map(m=>m.d))].sort().slice(0,3);list=list.filter(m=>days.includes(m.d)&&!m.score);}
  if(mode==='played')list=list.filter(m=>m.score);
  list.sort((x,y)=>x.d===y.d?(x.t<y.t?-1:1):(x.d<y.d?-1:1));
  document.getElementById('rows').innerHTML=list.length?list.map(row).join(''):'<p class="hint">No matches in this view.</p>';
  [['f-next','next'],['f-all','all'],['f-played','played']].forEach(([id,v])=>document.getElementById(id).classList.toggle('on',mode===v));
}
document.getElementById('f-next').onclick=()=>{mode='next';render();};
document.getElementById('f-all').onclick=()=>{mode='all';render();};
document.getElementById('f-played').onclick=()=>{mode='played';render();};
sel.onchange=render;
render();
</script></body></html>
"""


def main() -> None:
    rows, strategy = build_rows()
    html = (
        TEMPLATE
        .replace("__DATA__", json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        .replace("__STRATEGY__", strategy)
        .replace("__GENERATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    )
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(html):,} chars). Open it in any browser or attach it to a Claude chat.")


if __name__ == "__main__":
    main()
