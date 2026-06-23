from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "new/bridge_priorityD_robustness_dashboard_20260613"
HTMLS_DIR = ROOT / "htmls"

PATHS = {
    "pair": ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_pair_summary.csv",
    "radius": ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_radius_summary.csv",
    "lambda_range": ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_lambda_range_summary.csv",
    "lambda_alpha": ROOT / "new/bridge_group_trend_priorityD_lambda_response_20260613/csv/lambda_response_alpha_summary.csv",
    "lambda_pair": ROOT / "new/bridge_group_trend_priorityD_lambda_response_20260613/csv/lambda_response_pair_summary.csv",
    "integrated": ROOT / "new/bridge_priorityD_integrated_interpretation_20260613/csv/priorityD_pair_integrated_interpretation.csv",
    "changed": ROOT / "new/bridge_priorityD_integrated_interpretation_20260613/csv/what_changed_after_robustness.csv",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_value(value: Any) -> Any:
    if value is None:
        return "not available"
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "not available"
        return round(value, 6)
    if isinstance(value, (int, bool)):
        return value
    if pd.isna(value):
        return "not available"
    text = str(value)
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return "not available"
    return text


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def merge_inputs() -> dict[str, Any]:
    pair = read_csv(PATHS["pair"])
    radius = read_csv(PATHS["radius"])
    lambda_range = read_csv(PATHS["lambda_range"])
    lambda_alpha = read_csv(PATHS["lambda_alpha"])
    lambda_pair = read_csv(PATHS["lambda_pair"])
    integrated = read_csv(PATHS["integrated"])
    changed = read_csv(PATHS["changed"])

    if pair.empty or radius.empty or lambda_range.empty:
        missing = [name for name in ["pair", "radius", "lambda_range"] if read_csv(PATHS[name]).empty]
        raise RuntimeError(f"Missing required Priority D dashboard inputs: {missing}")

    if not integrated.empty:
        cols = [
            "pair_id",
            "priorityD_question",
            "primary_label",
            "secondary_flags",
            "support_scope",
            "classification_relation",
            "current_conclusion",
            "do_not_claim",
            "lambda_range_change_label",
            "lambda_range_change_text",
        ]
        pair = pair.merge(integrated[[c for c in cols if c in integrated.columns]], on="pair_id", how="left")

    if not changed.empty:
        cols = [
            "pair_id",
            "previous_reading",
            "robustness_update",
            "strengthened_interpretation",
            "weakened_or_softened_interpretation",
            "next_check",
        ]
        pair = pair.merge(changed[[c for c in cols if c in changed.columns]], on="pair_id", how="left")

    pair["delta_label"] = pair["delta_supported_perturbed_mean_vs_baseline_mean"].apply(classify_delta)
    pair["support_caveat"] = pair.apply(support_caveat, axis=1)
    pair["one_line_interpretation"] = pair.apply(one_line, axis=1)
    pair["display_label"] = pair["pair_id"].astype(str)

    group_summary = (
        pair.groupby(["target_priority", "group_name"], dropna=False)
        .agg(
            pair_count=("pair_id", "count"),
            baseline_eta20_mean=("baseline_eta20_mean", "mean"),
            supported_eta20_mean=("supported_perturbed_eta20_mean", "mean"),
            delta_mean=("delta_supported_perturbed_mean_vs_baseline_mean", "mean"),
            support_high_medium_fraction_mean=("perturbed_support_high_medium_fraction", "mean"),
            support_low_or_better_fraction_mean=("perturbed_support_low_or_better_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["target_priority", "group_name"])
    )

    meta = {
        "title": "Priority D Normal-Vector Robustness Dashboard",
        "pair_count": int(pair["pair_id"].nunique()),
        "total_jobs": int(pair["n_jobs"].sum()),
        "solver_success": int(pair["n_solver_success"].sum()),
        "mean_baseline_eta": float(pair["baseline_eta20_mean"].mean()),
        "mean_supported_eta": float(pair["supported_perturbed_eta20_mean"].mean()),
        "mean_supported_delta": float(pair["delta_supported_perturbed_mean_vs_baseline_mean"].mean()),
        "lambda_response_pair_count": int(lambda_pair["pair_id"].nunique()) if not lambda_pair.empty else 0,
        "notes": [
            "Radius 0.25 is treated as a local-neighborhood robustness probe.",
            "Larger sparse radii are off-path sensitivity probes, not sampled-manifold evidence.",
            "Lambda response is dynamics response on the same H(alpha) path; H eigenfeatures do not change with lambda.",
            "This dashboard is a guide for interpretation, not mechanism proof.",
        ],
    }
    return {
        "meta": {k: clean_value(v) for k, v in meta.items()},
        "pair_summary": clean_records(pair.sort_values(["target_priority", "group_name", "pair_id"])),
        "radius_summary": clean_records(radius.sort_values(["target_priority", "group_name", "pair_id", "radius"])),
        "lambda_range": clean_records(lambda_range.sort_values(["target_priority", "group_name", "pair_id", "radius"])),
        "lambda_alpha": clean_records(lambda_alpha.sort_values(["target_priority", "pair_id", "alpha"]) if not lambda_alpha.empty else lambda_alpha),
        "lambda_pair": clean_records(lambda_pair.sort_values(["target_priority", "pair_id"]) if not lambda_pair.empty else lambda_pair),
        "group_summary": clean_records(group_summary),
    }


def classify_delta(delta: Any) -> str:
    if pd.isna(delta):
        return "not available"
    delta = float(delta)
    if delta >= 0.20:
        return "raises eta strongly"
    if delta >= 0.05:
        return "raises eta"
    if delta <= -0.15:
        return "lowers eta strongly"
    if delta <= -0.05:
        return "lowers eta"
    return "near neutral"


def support_caveat(row: pd.Series) -> str:
    high = row.get("perturbed_support_high_medium_fraction")
    low = row.get("perturbed_support_low_or_better_fraction")
    max_low = row.get("max_radius_with_any_low_or_better")
    if pd.isna(high) or pd.isna(low) or pd.isna(max_low):
        return "support not available"
    if float(max_low) <= 0.25:
        scope = "local only to radius 0.25"
    elif float(max_low) <= 0.5:
        scope = "extends to radius 0.5"
    else:
        scope = f"extends to radius {float(max_low):g}"
    if float(high) < 0.10 and float(low) < 0.20:
        return f"{scope}; very support-limited"
    if float(high) < 0.20:
        return f"{scope}; support-limited"
    return scope


def one_line(row: pd.Series) -> str:
    conclusion = row.get("current_conclusion")
    if isinstance(conclusion, str) and conclusion and conclusion.lower() != "nan":
        return conclusion
    group = str(row.get("group_name", ""))
    delta = classify_delta(row.get("delta_supported_perturbed_mean_vs_baseline_mean"))
    if group == "lambda_dramatic":
        return "Straight path lambda response changes under local perturbation; treat as path-sensitive dynamics response."
    if group == "high_high_stable":
        return f"Functional stability candidate; local perturbation {delta}."
    if group == "low_high_transition":
        return "Supported perturbations raise eta, so endpoint transition and local-neighborhood sensitivity should be separated."
    if "valley" in group:
        return "Valley is currently a straight-path vulnerability candidate, not a robust bottleneck."
    return f"Context/control target; local perturbation {delta}."


def write_command_doc(html_path: Path, copy_path: Path) -> None:
    text = f"""# Priority D Robustness Dashboard Build

Generated files:

- `{html_path.relative_to(ROOT).as_posix()}`
- `{copy_path.relative_to(ROOT).as_posix()}`

Rebuild command:

```powershell
& 'C:\\Users\\User\\anaconda3\\envs\\py311-cu132\\python.exe' new/build_priorityD_robustness_dashboard.py
```

Inputs:

- `new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_pair_summary.csv`
- `new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_radius_summary.csv`
- `new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_lambda_range_summary.csv`
- `new/bridge_group_trend_priorityD_lambda_response_20260613/csv/lambda_response_alpha_summary.csv`
- `new/bridge_group_trend_priorityD_lambda_response_20260613/csv/lambda_response_pair_summary.csv`
- `new/bridge_priorityD_integrated_interpretation_20260613/csv/priorityD_pair_integrated_interpretation.csv`
- `new/bridge_priorityD_integrated_interpretation_20260613/csv/what_changed_after_robustness.csv`
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Priority D Robustness Dashboard</title>
<style>
:root{--ink:#182033;--muted:#5d6879;--line:#d8dee9;--soft:#f5f7fb;--paper:#fff;--blue:#245b9f;--green:#14745f;--orange:#b25b13;--red:#a73345;--violet:#6d4cc2}
*{box-sizing:border-box} body{margin:0;color:var(--ink);background:#fff;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;padding:28px 32px 20px;border-bottom:1px solid var(--line);background:#fbfcfe;position:sticky;top:0;z-index:5}
h1{margin:0 0 7px;font-size:clamp(26px,3vw,38px);letter-spacing:0} h2{margin:0 0 12px;font-size:18px} h3{margin:0 0 8px;font-size:14px;color:#334155}
p{margin:0}.eyebrow{margin:0 0 4px;color:#0f6d7a;font-weight:760;font-size:13px;text-transform:uppercase}.lead{max-width:1080px;color:var(--muted);font-size:16px}
nav{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;min-width:310px} nav a{color:#0f6d7a;text-decoration:none;border:1px solid #b7d6dc;background:#f0fbfd;padding:7px 10px;border-radius:6px;font-size:13px}
main{padding:22px 32px 44px;max-width:1740px}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-bottom:18px}
.metric{border:1px solid var(--line);border-radius:7px;padding:11px 12px;background:#fff;min-height:78px}.metric b{display:block;color:var(--muted);font-size:12px;font-weight:650;margin-bottom:3px}.metric span{display:block;font-size:23px;font-weight:760}.metric small{color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px;border:1px solid var(--line);background:var(--soft);border-radius:7px;margin-bottom:16px}
label{color:var(--muted);font-size:13px;font-weight:650} select{margin-left:7px;min-width:260px;padding:7px 9px;border:1px solid #cbd3df;border-radius:6px;background:#fff;color:var(--ink)}
.tabs{display:flex;gap:8px;flex-wrap:wrap}.tabs button{border:1px solid #cbd3df;background:#fff;color:#334155;padding:8px 11px;border-radius:6px;cursor:pointer}.tabs button.active{border-color:#8bc4ce;background:#effbfc;color:#0f6d7a;font-weight:750}
.note{border:1px solid var(--line);background:#fff;border-radius:7px;padding:12px 14px;color:var(--muted);margin-bottom:16px}.note strong{color:var(--ink)}
.dashboard-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}.panel{border:1px solid var(--line);background:#fff;border-radius:7px;padding:14px;overflow:hidden}.span-2{grid-column:span 2}
.chart svg{width:100%;height:auto;display:block}.axis{stroke:#8b95a7;stroke-width:1}.gridline{stroke:#e8ecf2;stroke-width:1}.tick{fill:#5b6576;font-size:11px}.label{fill:#374151;font-size:12px;font-weight:650}.caption{color:var(--muted);font-size:12px;margin-top:8px}
.legend{display:flex;flex-wrap:wrap;gap:8px 14px;color:var(--muted);font-size:12px;margin-top:8px}.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.table-wrap{overflow:auto;max-height:520px;border:1px solid #edf0f5;border-radius:6px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid #e7ebf1;padding:7px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child,td:nth-child(2),th:nth-child(2){text-align:left}th{background:#eef2f7;color:#394457;position:sticky;top:0}
.pair-card{border:1px solid #e2e7ee;background:#fff;border-radius:7px;padding:12px;margin-bottom:10px}.pair-title{display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap}.pair-title b{font-size:15px}.pill{display:inline-flex;align-items:center;border:1px solid #cbd5e1;background:#f8fafc;border-radius:999px;padding:3px 8px;font-size:12px;color:#334155}.pill.local{border-color:#b7d6dc;background:#effbfc;color:#0f6d7a}.pill.sparse{border-color:#f1c48b;background:#fff8ed;color:#8a4b10}.delta-pos{color:var(--green);font-weight:750}.delta-neg{color:var(--red);font-weight:750}.delta-neu{color:#5d6879;font-weight:750}
.section-title{margin:28px 0 12px;padding-top:6px;border-top:1px solid var(--line)}
@media(max-width:980px){header{display:block;padding:22px 18px 16px;position:static}nav{justify-content:flex-start;margin-top:14px;min-width:0}main{padding:18px}.dashboard-grid{grid-template-columns:1fr}.span-2{grid-column:span 1}select{min-width:190px;max-width:100%}}
</style>
</head>
<body>
<header>
  <div>
    <p class="eyebrow">Priority D robustness review</p>
    <h1>Normal-Vector Robustness Dashboard</h1>
    <p class="lead">Straight bridge path baseline과 normal-direction local perturbation을 비교한다. Radius 0.25는 local-neighborhood 후보, support가 sparse한 큰 radius는 off-path sensitivity probe로만 본다.</p>
  </div>
  <nav>
    <a href="54_priority6_eigenstate_dashboard.html">54 eigenstate</a>
    <a href="50_grouped_bridge_validity_dashboard.html">50 bridge validity</a>
    <a href="53_all_pair_eigenstate_screening_dashboard.html">53 all-pair eigen</a>
  </nav>
</header>
<main>
  <section class="summary-grid" id="summaryCards"></section>
  <section class="controls">
    <div class="tabs" id="priorityTabs"></div>
    <label>Pair <select id="pairSelect"></select></label>
  </section>
  <section class="note" id="priorityNote"></section>
  <section class="dashboard-grid">
    <article class="panel span-2"><h2>1. Baseline vs Supported Perturbation</h2><div class="chart" id="baselineChart"></div><p class="caption">Each pair compares straight path eta20 mean at radius 0 against supported perturbed eta20 mean. Positive delta means nearby supported perturbations are higher than the straight-path baseline.</p></article>
    <article class="panel"><h2>2. Selected Pair: Eta Distribution by Radius</h2><div class="chart" id="radiusEtaChart"></div><div class="legend"><span><i class="swatch" style="background:#245b9f"></i>q50 eta20</span><span><i class="swatch" style="background:#9fbce5"></i>q10-q90 band</span></div></article>
    <article class="panel"><h2>3. Selected Pair: Support Fraction by Radius</h2><div class="chart" id="supportChart"></div><div class="legend"><span><i class="swatch" style="background:#14745f"></i>high/medium</span><span><i class="swatch" style="background:#b25b13"></i>low or better</span></div></article>
    <article class="panel"><h2>4. Selected Pair: Lambda Eta-Range by Radius</h2><div class="chart" id="lambdaRangeChart"></div><p class="caption">This is eta spread across lambda values on the same H(alpha) path. It is not a change in H eigenfeatures.</p></article>
    <article class="panel"><h2>5. Pair Interpretation Card</h2><div id="pairCard"></div></article>
    <article class="panel span-2"><h2>6. Priority / Category Summary</h2><div class="table-wrap" id="groupSummaryTable"></div></article>
    <article class="panel span-2"><h2>7. Pair Table</h2><div class="table-wrap" id="pairTable"></div></article>
    <article class="panel span-2"><h2>8. Lambda-Response Detail When Available</h2><div class="chart" id="lambdaAlphaChart"></div><p class="caption">Only pairs included in the lambda-response postprocess have alpha-level lambda detail. Missing rows mean not stored for this pair, not a browser error.</p></article>
  </section>
  <section id="prioritySections"></section>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const COLORS = {baseline:'#245b9f', supported:'#14745f', delta:'#6d4cc2', q:'#9fbce5', orange:'#b25b13', red:'#a73345', muted:'#5d6879'};
const priorityNames = {
  1:'Priority 1 · lambda dramatic',
  2:'Priority 2 · stable / transition',
  3:'Priority 3 · endpoint context',
  4:'Priority 4 · valley vulnerability'
};
let state = {priority:'all', pairId:DATA.pair_summary[0]?.pair_id || ''};
function num(v){ const n=Number(v); return Number.isFinite(n)?n:null; }
function fmt(v,d=3){ const n=num(v); if(n===null) return 'not available'; return n.toFixed(d); }
function pct(v){ const n=num(v); if(n===null) return 'not available'; return Math.round(n*100)+'%'; }
function esc(s){ return String(s ?? 'not available').replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
function shortText(s, n=230){ s=String(s||'not available'); return s.length>n ? s.slice(0,n-1)+'…' : s; }
function clsDelta(v){ const n=num(v); if(n===null) return 'delta-neu'; if(n>0.05) return 'delta-pos'; if(n<-0.05) return 'delta-neg'; return 'delta-neu'; }
function filteredPairs(){ return DATA.pair_summary.filter(d=>state.priority==='all' || String(d.target_priority)===String(state.priority)); }
function currentPair(){ return DATA.pair_summary.find(d=>d.pair_id===state.pairId) || filteredPairs()[0] || DATA.pair_summary[0]; }
function setPairOptions(){
  const pairs=filteredPairs();
  if(!pairs.find(d=>d.pair_id===state.pairId)) state.pairId=pairs[0]?.pair_id || '';
  document.getElementById('pairSelect').innerHTML=pairs.map(d=>`<option value="${esc(d.pair_id)}"${d.pair_id===state.pairId?' selected':''}>${esc(d.display_label)}</option>`).join('');
}
function renderSummary(){
 const m=DATA.meta;
 const cards=[
  ['pairs', m.pair_count, 'Priority D targets'],
  ['solver success', `${m.solver_success}/${m.total_jobs}`, 'all completed jobs'],
  ['mean baseline eta20', fmt(m.mean_baseline_eta), 'straight path'],
  ['mean supported eta20', fmt(m.mean_supported_eta), 'supported perturbation'],
  ['mean delta', fmt(m.mean_supported_delta), 'supported - baseline'],
  ['lambda detail pairs', m.lambda_response_pair_count, 'postprocessed pairs']
 ];
 document.getElementById('summaryCards').innerHTML=cards.map(c=>`<div class="metric"><b>${esc(c[0])}</b><span>${esc(c[1])}</span><small>${esc(c[2])}</small></div>`).join('');
}
function renderTabs(){
 const ids=['all',1,2,3,4];
 document.getElementById('priorityTabs').innerHTML=ids.map(id=>`<button type="button" class="${String(state.priority)===String(id)?'active':''}" data-priority="${id}">${id==='all'?'All priorities':priorityNames[id]}</button>`).join('');
 document.querySelectorAll('#priorityTabs button').forEach(btn=>btn.addEventListener('click',()=>{state.priority=btn.dataset.priority; setPairOptions(); renderAll();}));
 document.getElementById('pairSelect').onchange=e=>{state.pairId=e.target.value; renderAll();};
}
function renderPriorityNote(){
 const p=state.priority;
 const notes={
  all:'Priority 1-4를 한 페이지에서 비교한다. Pair 선택 시 오른쪽 plot은 해당 pair만 보여 준다.',
  1:'Lambda dramatic: straight path의 lambda response가 local perturbation에서도 유지되는지 보는 검증이다. H eigenfeature 변화로 해석하지 않는다.',
  2:'High-high stable과 low-high transition: eigenstate invariance와 functional stability, endpoint-driven change와 local sensitivity를 분리해서 본다.',
  3:'Endpoint context: main mechanism proof가 아니라 endpoint type과 support caveat를 보정하는 control 성격이다.',
  4:'Valley vulnerability: 현재는 robust bottleneck이 아니라 straight-path vulnerability 후보로 표시한다.'
 };
 document.getElementById('priorityNote').innerHTML=`<strong>${esc(p==='all'?'All priorities':priorityNames[p])}</strong><br>${esc(notes[p])}`;
}
function svgWrap(w,h,body){ return `<svg viewBox="0 0 ${w} ${h}" role="img">${body}</svg>`; }
function scale(v,min,max,a,b){ if(max===min) return (a+b)/2; return a+(v-min)*(b-a)/(max-min); }
function axes(w,h,pad,yLabel,xLabel){
 let g=`<line class="axis" x1="${pad.l}" y1="${h-pad.b}" x2="${w-pad.r}" y2="${h-pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h-pad.b}"/>`;
 g+=`<text class="label" x="${pad.l}" y="${pad.t-10}">${esc(yLabel)}</text><text class="label" x="${w-pad.r}" y="${h-8}" text-anchor="end">${esc(xLabel)}</text>`;
 for(let i=0;i<=4;i++){ const y=scale(i,0,4,h-pad.b,pad.t); g+=`<line class="gridline" x1="${pad.l}" y1="${y}" x2="${w-pad.r}" y2="${y}"/><text class="tick" x="${pad.l-8}" y="${y+4}" text-anchor="end">${(i/4).toFixed(2)}</text>`; }
 return g;
}
function renderBaselineChart(){
 const rows=filteredPairs();
 const w=1180,h=Math.max(320, rows.length*34+80), pad={l:210,r:32,t:24,b:38};
 let body=axes(w,h,pad,'eta20 mean','baseline / supported perturbation');
 const yStep=(h-pad.t-pad.b)/Math.max(rows.length,1);
 rows.forEach((d,i)=>{
   const y=pad.t+i*yStep+yStep/2;
   const b=num(d.baseline_eta20_mean)||0, s=num(d.supported_perturbed_eta20_mean)||0;
   body+=`<text class="tick" x="${pad.l-8}" y="${y+4}" text-anchor="end">${esc(d.pair_id)}</text>`;
   body+=`<line x1="${scale(b,0,1,pad.l,w-pad.r)}" y1="${y}" x2="${scale(s,0,1,pad.l,w-pad.r)}" y2="${y}" stroke="${COLORS.delta}" stroke-width="2" opacity=".55"/>`;
   body+=`<circle cx="${scale(b,0,1,pad.l,w-pad.r)}" cy="${y}" r="5" fill="${COLORS.baseline}"><title>baseline ${fmt(b)}</title></circle>`;
   body+=`<circle cx="${scale(s,0,1,pad.l,w-pad.r)}" cy="${y}" r="5" fill="${COLORS.supported}"><title>supported ${fmt(s)}</title></circle>`;
 });
 body+=`<g transform="translate(${pad.l},${h-20})"><circle r="5" fill="${COLORS.baseline}"/><text class="tick" x="10" y="4">baseline</text><circle cx="95" r="5" fill="${COLORS.supported}"/><text class="tick" x="105" y="4">supported</text></g>`;
 document.getElementById('baselineChart').innerHTML=svgWrap(w,h,body);
}
function linePath(points){ return points.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' '); }
function renderRadiusEta(){
 const pair=currentPair(); const rows=DATA.radius_summary.filter(d=>d.pair_id===pair.pair_id).sort((a,b)=>num(a.radius)-num(b.radius));
 const w=760,h=360,pad={l:54,r:28,t:24,b:48};
 let body=axes(w,h,pad,'eta20','radius');
 const radii=rows.map(d=>num(d.radius)||0), maxR=Math.max(...radii,1);
 if(rows.length){
  const bandTop=rows.map(d=>[scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), scale(num(d.eta20_q90)||0,0,1,h-pad.b,pad.t)]);
  const bandBot=rows.slice().reverse().map(d=>[scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), scale(num(d.eta20_q10)||0,0,1,h-pad.b,pad.t)]);
  body+=`<path d="${linePath(bandTop)} ${linePath(bandBot).replace('M','L')} Z" fill="${COLORS.q}" opacity=".35"/>`;
  const med=rows.map(d=>[scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), scale(num(d.eta20_q50)||0,0,1,h-pad.b,pad.t)]);
  body+=`<path d="${linePath(med)}" fill="none" stroke="${COLORS.baseline}" stroke-width="3"/>`;
  rows.forEach(d=>{const x=scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), y=scale(num(d.eta20_q50)||0,0,1,h-pad.b,pad.t); body+=`<circle cx="${x}" cy="${y}" r="5" fill="${COLORS.baseline}"><title>r=${fmt(d.radius,2)} q50=${fmt(d.eta20_q50)}</title></circle><text class="tick" x="${x}" y="${h-pad.b+18}" text-anchor="middle">${fmt(d.radius,2)}</text>`;});
 }
 document.getElementById('radiusEtaChart').innerHTML=svgWrap(w,h,body);
}
function renderSupport(){
 const pair=currentPair(); const rows=DATA.radius_summary.filter(d=>d.pair_id===pair.pair_id).sort((a,b)=>num(a.radius)-num(b.radius));
 const w=760,h=360,pad={l:54,r:28,t:24,b:50}; let body=axes(w,h,pad,'fraction','radius');
 const bw=(w-pad.l-pad.r)/Math.max(rows.length*2.5,1);
 rows.forEach((d,i)=>{
   const x=pad.l+(i+0.7)*(w-pad.l-pad.r)/Math.max(rows.length,1);
   const high=num(d.support_high_medium_fraction)||0, low=num(d.support_low_or_better_fraction)||0;
   const yH=scale(high,0,1,h-pad.b,pad.t), yL=scale(low,0,1,h-pad.b,pad.t);
   body+=`<rect x="${x-bw}" y="${yH}" width="${bw}" height="${h-pad.b-yH}" fill="${COLORS.green}"><title>high/medium ${pct(high)}</title></rect>`;
   body+=`<rect x="${x+2}" y="${yL}" width="${bw}" height="${h-pad.b-yL}" fill="${COLORS.orange}"><title>low or better ${pct(low)}</title></rect>`;
   body+=`<text class="tick" x="${x}" y="${h-pad.b+18}" text-anchor="middle">${fmt(d.radius,2)}</text>`;
 });
 document.getElementById('supportChart').innerHTML=svgWrap(w,h,body);
}
function renderLambdaRange(){
 const pair=currentPair(); const rows=DATA.lambda_range.filter(d=>d.pair_id===pair.pair_id).sort((a,b)=>num(a.radius)-num(b.radius));
 const w=760,h=360,pad={l:54,r:28,t:24,b:50}; let maxY=Math.max(...rows.flatMap(d=>[num(d.lambda_range_max)||0,num(d.lambda_range_q90)||0,num(d.lambda_range_median)||0]),0.1);
 let body=`<line class="axis" x1="${pad.l}" y1="${h-pad.b}" x2="${w-pad.r}" y2="${h-pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h-pad.b}"/><text class="label" x="${pad.l}" y="${pad.t-10}">eta range across lambda</text><text class="label" x="${w-pad.r}" y="${h-8}" text-anchor="end">radius</text>`;
 for(let i=0;i<=4;i++){const val=maxY*i/4,y=scale(val,0,maxY,h-pad.b,pad.t);body+=`<line class="gridline" x1="${pad.l}" y1="${y}" x2="${w-pad.r}" y2="${y}"/><text class="tick" x="${pad.l-8}" y="${y+4}" text-anchor="end">${val.toFixed(2)}</text>`;}
 const maxR=Math.max(...rows.map(d=>num(d.radius)||0),1);
 const med=rows.map(d=>[scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), scale(num(d.lambda_range_median)||0,0,maxY,h-pad.b,pad.t)]);
 const q90=rows.map(d=>[scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), scale(num(d.lambda_range_q90)||0,0,maxY,h-pad.b,pad.t)]);
 body+=`<path d="${linePath(q90)}" fill="none" stroke="${COLORS.orange}" stroke-width="2" stroke-dasharray="5 4"/><path d="${linePath(med)}" fill="none" stroke="${COLORS.violet}" stroke-width="3"/>`;
 rows.forEach(d=>{const x=scale(num(d.radius)||0,0,maxR,pad.l,w-pad.r), y=scale(num(d.lambda_range_median)||0,0,maxY,h-pad.b,pad.t); body+=`<circle cx="${x}" cy="${y}" r="5" fill="${COLORS.violet}"><title>r=${fmt(d.radius,2)} median=${fmt(d.lambda_range_median)}</title></circle><text class="tick" x="${x}" y="${h-pad.b+18}" text-anchor="middle">${fmt(d.radius,2)}</text>`;});
 body+=`<g transform="translate(${pad.l},${h-20})"><line x1="0" y1="0" x2="24" y2="0" stroke="${COLORS.violet}" stroke-width="3"/><text class="tick" x="30" y="4">median</text><line x1="98" y1="0" x2="122" y2="0" stroke="${COLORS.orange}" stroke-width="2" stroke-dasharray="5 4"/><text class="tick" x="128" y="4">q90</text></g>`;
 document.getElementById('lambdaRangeChart').innerHTML=svgWrap(w,h,body);
}
function renderPairCard(){
 const d=currentPair();
 document.getElementById('pairCard').innerHTML=`<div class="pair-card">
  <div class="pair-title"><b>${esc(d.pair_id)}</b><span class="pill local">${esc(d.target_category)}</span></div>
  <p><strong>Baseline eta20 mean:</strong> ${fmt(d.baseline_eta20_mean)} &nbsp; <strong>Supported eta20 mean:</strong> ${fmt(d.supported_perturbed_eta20_mean)} &nbsp; <strong>Delta:</strong> <span class="${clsDelta(d.delta_supported_perturbed_mean_vs_baseline_mean)}">${fmt(d.delta_supported_perturbed_mean_vs_baseline_mean)}</span></p>
  <p><strong>Support caveat:</strong> ${esc(d.support_caveat)}</p>
  <p><strong>Interpretation:</strong> ${esc(d.one_line_interpretation)}</p>
  <p><strong>Strengthened:</strong> ${esc(shortText(d.strengthened_interpretation,180))}</p>
  <p><strong>Softened:</strong> ${esc(shortText(d.weakened_or_softened_interpretation,180))}</p>
 </div>`;
}
function table(rows, cols){
 if(!rows.length) return '<p class="caption">not stored for this selection</p>';
 return `<table><thead><tr>${cols.map(c=>`<th>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${c.fmt?c.fmt(r[c.key],r):esc(r[c.key])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderTables(){
 const pairs=filteredPairs();
 document.getElementById('groupSummaryTable').innerHTML=table(DATA.group_summary.filter(d=>state.priority==='all'||String(d.target_priority)===String(state.priority)),[
  {key:'target_priority',label:'priority'},{key:'group_name',label:'group'},{key:'pair_count',label:'pairs'},
  {key:'baseline_eta20_mean',label:'baseline mean',fmt:v=>fmt(v)},{key:'supported_eta20_mean',label:'supported mean',fmt:v=>fmt(v)},
  {key:'delta_mean',label:'delta mean',fmt:v=>fmt(v)},{key:'support_high_medium_fraction_mean',label:'high/medium',fmt:v=>pct(v)},{key:'support_low_or_better_fraction_mean',label:'low+',fmt:v=>pct(v)}
 ]);
 document.getElementById('pairTable').innerHTML=table(pairs,[
  {key:'pair_id',label:'pair'},{key:'target_priority',label:'priority'},{key:'group_name',label:'group'},{key:'target_category',label:'target category'},
  {key:'baseline_eta20_mean',label:'baseline',fmt:v=>fmt(v)},{key:'supported_perturbed_eta20_mean',label:'supported',fmt:v=>fmt(v)},
  {key:'delta_supported_perturbed_mean_vs_baseline_mean',label:'delta',fmt:v=>`<span class="${clsDelta(v)}">${fmt(v)}</span>`},
  {key:'support_caveat',label:'support caveat'},{key:'one_line_interpretation',label:'interpretation',fmt:v=>esc(shortText(v,130))}
 ]);
}
function renderLambdaAlpha(){
 const pair=currentPair();
 const rows=DATA.lambda_alpha.filter(d=>d.pair_id===pair.pair_id).sort((a,b)=>num(a.alpha)-num(b.alpha));
 if(!rows.length){ document.getElementById('lambdaAlphaChart').innerHTML='<p class="caption">not stored for this pair</p>'; return; }
 const w=1180,h=330,pad={l:54,r:28,t:24,b:50}; const maxY=Math.max(...rows.map(d=>num(d.eta20_range_across_lambda)||0),0.1);
 let body=`<line class="axis" x1="${pad.l}" y1="${h-pad.b}" x2="${w-pad.r}" y2="${h-pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h-pad.b}"/><text class="label" x="${pad.l}" y="${pad.t-10}">eta range across lambda</text><text class="label" x="${w-pad.r}" y="${h-8}" text-anchor="end">alpha</text>`;
 for(let i=0;i<=4;i++){const val=maxY*i/4,y=scale(val,0,maxY,h-pad.b,pad.t);body+=`<line class="gridline" x1="${pad.l}" y1="${y}" x2="${w-pad.r}" y2="${y}"/><text class="tick" x="${pad.l-8}" y="${y+4}" text-anchor="end">${val.toFixed(2)}</text>`;}
 const pts=rows.map(d=>[scale(num(d.alpha)||0,0,1,pad.l,w-pad.r), scale(num(d.eta20_range_across_lambda)||0,0,maxY,h-pad.b,pad.t)]);
 body+=`<path d="${linePath(pts)}" fill="none" stroke="${COLORS.red}" stroke-width="3"/>`;
 rows.forEach(d=>{const x=scale(num(d.alpha)||0,0,1,pad.l,w-pad.r), y=scale(num(d.eta20_range_across_lambda)||0,0,maxY,h-pad.b,pad.t); body+=`<circle cx="${x}" cy="${y}" r="5" fill="${COLORS.red}"><title>alpha=${fmt(d.alpha,2)} range=${fmt(d.eta20_range_across_lambda)} best lambda=${esc(d.best_lambda)}</title></circle><text class="tick" x="${x}" y="${h-pad.b+18}" text-anchor="middle">${fmt(d.alpha,2)}</text>`;});
 document.getElementById('lambdaAlphaChart').innerHTML=svgWrap(w,h,body);
}
function renderPrioritySections(){
 const groups=[1,2,3,4].map(p=>DATA.pair_summary.filter(d=>String(d.target_priority)===String(p)));
 document.getElementById('prioritySections').innerHTML=groups.map((rows,i)=>{
   const p=i+1;
   return `<section class="section-title"><h2>${esc(priorityNames[p])}</h2>${rows.map(d=>`<div class="pair-card"><div class="pair-title"><b>${esc(d.pair_id)}</b><span class="pill">${esc(d.target_category)}</span></div><p>baseline ${fmt(d.baseline_eta20_mean)} → supported ${fmt(d.supported_perturbed_eta20_mean)} <span class="${clsDelta(d.delta_supported_perturbed_mean_vs_baseline_mean)}">delta ${fmt(d.delta_supported_perturbed_mean_vs_baseline_mean)}</span></p><p>${esc(shortText(d.one_line_interpretation,260))}</p><p class="caption">${esc(d.support_caveat)}</p></div>`).join('')}</section>`;
 }).join('');
}
function renderAll(){
 renderTabs(); setPairOptions(); renderPriorityNote(); renderBaselineChart(); renderRadiusEta(); renderSupport(); renderLambdaRange(); renderPairCard(); renderTables(); renderLambdaAlpha(); renderPrioritySections();
}
renderSummary(); renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTMLS_DIR.mkdir(parents=True, exist_ok=True)
    data = merge_inputs()
    json_payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA__", json_payload.replace("</", "<\\/"))
    html_path = OUT_DIR / "priorityD_robustness_dashboard.html"
    copy_path = HTMLS_DIR / "55_priorityD_robustness_dashboard.html"
    html_path.write_text(html, encoding="utf-8")
    shutil.copyfile(html_path, copy_path)
    write_command_doc(html_path, copy_path)
    print(f"wrote {html_path}")
    print(f"wrote {copy_path}")


if __name__ == "__main__":
    main()
