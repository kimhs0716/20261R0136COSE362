from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "new/bridge_normal_geometry_high_high_results_20260614/csv"
CONCLUSIONS = ROOT / "new/bridge_normal_geometry_high_high_conclusions_20260614/csv"
RUN = ROOT / "outputs/bridge_normal_geometry_high_high_core_lambda35_20260614/csv"
OUT_DIR = ROOT / "new/bridge_normal_geometry_dashboard_20260614"
HTMLS_DIR = ROOT / "htmls"

HTML_NAME = "56_high_high_normal_geometry_dashboard.html"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(value: Any) -> Any:
    if value is None:
        return "not available"
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
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


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def build_payload() -> dict[str, Any]:
    radius = read_csv(RESULTS / "radius_support_eta_summary.csv")
    labels = read_csv(RESULTS / "geometry_label_counts.csv")
    label_lambda = read_csv(RESULTS / "geometry_label_by_dominant_best_lambda.csv")
    pair = read_csv(RESULTS / "pair_geometry_summary.csv")
    pair_alpha = read_csv(RESULTS / "pair_alpha_geometry_metrics.csv")
    claims = read_csv(CONCLUSIONS / "normal_geometry_claim_evidence_caveat.csv")
    label_interp = read_csv(CONCLUSIONS / "geometry_label_interpretation.csv")
    next_actions = read_csv(CONCLUSIONS / "normal_geometry_next_actions.csv")
    detail = read_csv(RUN / "normal_robustness_detail.csv")

    required = {
        "radius": radius,
        "labels": labels,
        "label_lambda": label_lambda,
        "pair": pair,
        "pair_alpha": pair_alpha,
        "claims": claims,
    }
    missing = [name for name, df in required.items() if df.empty]
    if missing:
        raise RuntimeError(f"Missing dashboard input(s): {missing}")

    pair_radius = summarize_pair_radius(detail)
    pair_alpha_label_counts = (
        pair_alpha[pair_alpha["baseline_is_high"].astype(bool)]["geometry_label_pair_alpha"]
        .value_counts()
        .rename_axis("pair_alpha_geometry_label")
        .reset_index(name="n_pair_alpha")
    )
    pair["display_label"] = pair["pair_id"].astype(str)
    pair["label_short"] = pair["pair_geometry_label"].map(short_label)
    label_order = labels["pair_geometry_label"].tolist()

    meta = {
        "title": "High-High Normal Geometry Dashboard",
        "pair_count": int(pair["pair_id"].nunique()),
        "simulation_count": 48276,
        "lambda": 35,
        "radius_count": int(radius["radius"].nunique()),
        "alpha_point_count": int(pair_alpha[["pair_id", "alpha"]].drop_duplicates().shape[0]),
        "key_message": (
            "lambda=35에서 실제 high eta인 alpha point는 대체로 normal 방향에서도 일정 폭을 보이지만, "
            "많은 pair는 lambda=35 straight path가 best-lambda high-eta behavior를 대표하지 못한다."
        ),
    }
    return {
        "meta": clean(meta),
        "radius": records(radius),
        "labels": records(labels),
        "label_lambda": records(label_lambda),
        "pair": records(pair.sort_values(["pair_geometry_label", "pair_id"])),
        "pair_alpha": records(pair_alpha.sort_values(["pair_id", "alpha"])),
        "pair_radius": records(pair_radius.sort_values(["pair_id", "radius"])),
        "claims": records(claims),
        "label_interpretation": records(label_interp),
        "next_actions": records(next_actions),
        "pair_alpha_label_counts": records(pair_alpha_label_counts),
        "label_order": label_order,
    }


def short_label(label: str) -> str:
    mapping = {
        "path_not_centered_or_lambda35_mismatch_candidate": "path/lambda mismatch",
        "supported_high_eta_tube_or_cloud_candidate": "supported tube/cloud",
        "supported_tube_candidate": "supported tube",
        "lambda35_low_but_retention_supported": "lambda35 low, retained",
    }
    return mapping.get(str(label), str(label))


def summarize_pair_radius(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    d = detail[detail["solver_success"].fillna(False)].copy()
    d["radius"] = pd.to_numeric(d["radius"], errors="coerce")
    d["eta20"] = pd.to_numeric(d["eta20"], errors="coerce")
    d["support_level"] = d["support_level"].astype(str)
    g = (
        d.groupby(["pair_id", "radius"], as_index=False)
        .agg(
            eta20_median=("eta20", "median"),
            eta20_q10=("eta20", lambda x: x.quantile(0.10)),
            eta20_q25=("eta20", lambda x: x.quantile(0.25)),
            eta20_q75=("eta20", lambda x: x.quantile(0.75)),
            eta20_q90=("eta20", lambda x: x.quantile(0.90)),
            high_eta_fraction=("eta20", lambda x: float((x >= 0.75).mean())),
            support_high_medium_fraction=("support_level", lambda x: float(x.isin(["high", "medium"]).mean())),
            support_low_or_better_fraction=("support_level", lambda x: float(x.isin(["high", "medium", "low"]).mean())),
            sparse_fraction=("support_level", lambda x: float(x.isin(["sparse"]).mean())),
            nearest_support_distance_median=("nearest_support_distance", "median"),
        )
    )
    return g


def html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>High-High Normal Geometry Dashboard</title>
<style>
:root{{--ink:#172033;--muted:#5f6978;--line:#d8dee8;--soft:#f5f7fa;--paper:#fff;--blue:#245b9f;--teal:#14745f;--amber:#b25b13;--red:#a33a48;--violet:#6d4cc2;--slate:#526071}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:#fff;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}
header{{padding:26px 32px 20px;border-bottom:1px solid var(--line);background:#fbfcfe;position:sticky;top:0;z-index:5}}
.top{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}} h1{{margin:0 0 8px;font-size:clamp(27px,3vw,40px);letter-spacing:0}} h2{{margin:0 0 12px;font-size:18px}} h3{{margin:0 0 8px;font-size:14px;color:#354155}} p{{margin:0}}
.eyebrow{{color:#0f6d7a;font-weight:760;font-size:13px;text-transform:uppercase;margin-bottom:5px}} .lead{{max-width:1120px;color:var(--muted);font-size:16px}}
nav{{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;min-width:340px}} nav a{{color:#0f6d7a;text-decoration:none;border:1px solid #b7d6dc;background:#effbfc;padding:7px 10px;border-radius:6px;font-size:13px}}
main{{padding:22px 32px 44px;max-width:1760px}} .summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-bottom:16px}}
.metric{{border:1px solid var(--line);border-radius:7px;padding:11px 12px;background:#fff;min-height:78px}} .metric b{{display:block;color:var(--muted);font-size:12px;font-weight:650;margin-bottom:3px}} .metric span{{display:block;font-size:23px;font-weight:760}} .metric small{{color:var(--muted)}}
.note{{border:1px solid var(--line);background:#fff;border-radius:7px;padding:12px 14px;color:var(--muted);margin-bottom:16px}} .note strong{{color:var(--ink)}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px;border:1px solid var(--line);background:var(--soft);border-radius:7px;margin-bottom:16px}} label{{color:var(--muted);font-size:13px;font-weight:650}} select{{margin-left:7px;min-width:290px;max-width:100%;padding:7px 9px;border:1px solid #cbd3df;border-radius:6px;background:#fff;color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}} .panel{{border:1px solid var(--line);background:#fff;border-radius:7px;padding:14px;overflow:hidden}} .span2{{grid-column:span 2}}
.chart svg{{width:100%;height:auto;display:block}} .axis{{stroke:#8b95a7;stroke-width:1}} .gridline{{stroke:#e8ecf2;stroke-width:1}} .tick{{fill:#5b6576;font-size:11px}} .label{{fill:#374151;font-size:12px;font-weight:650}}
.caption{{color:var(--muted);font-size:12px;margin-top:8px}} .legend{{display:flex;flex-wrap:wrap;gap:8px 14px;color:var(--muted);font-size:12px;margin-top:8px}} .swatch{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.claim-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;margin-bottom:16px}} .claim{{border:1px solid #dfe5ee;border-left:4px solid var(--blue);border-radius:7px;padding:12px;background:#fff}} .claim b{{display:block;margin-bottom:6px}} .claim small{{color:var(--muted)}} .claim p{{margin-top:6px;color:#445066;font-size:13px}}
.pill{{display:inline-flex;align-items:center;border:1px solid #cbd5e1;background:#f8fafc;border-radius:999px;padding:3px 8px;font-size:12px;color:#334155}} .pill.blue{{border-color:#9fbce5;background:#f2f7ff;color:#245b9f}} .pill.green{{border-color:#a7d7ca;background:#effaf6;color:#14745f}} .pill.amber{{border-color:#f0c88c;background:#fff8ed;color:#8a4b10}} .pill.red{{border-color:#e4a6af;background:#fff5f6;color:#a33a48}}
.table-wrap{{overflow:auto;max-height:520px;border:1px solid #edf0f5;border-radius:6px}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid #e7ebf1;padding:7px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} th{{background:#eef2f7;color:#394457;position:sticky;top:0}}
.pair-card{{border:1px solid #e2e7ee;background:#fff;border-radius:7px;padding:12px}} .pair-title{{display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap}} .pair-title b{{font-size:15px}} .kv{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}} .kv div{{border:1px solid #edf0f5;border-radius:6px;padding:8px}} .kv b{{display:block;color:var(--muted);font-size:11px}} .kv span{{font-weight:750}}
@media(max-width:980px){{header{{position:static;padding:22px 18px}} .top{{display:block}} nav{{justify-content:flex-start;margin-top:14px;min-width:0}} main{{padding:18px}} .grid{{grid-template-columns:1fr}} .span2{{grid-column:span 1}} .kv{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <div class="top">
    <div>
      <p class="eyebrow">High-high normal geometry</p>
      <h1>Normal-Direction Geometry Dashboard</h1>
      <p class="lead">D/S same-S high-high core 103개 pair의 lambda=35 normal-direction sweep 결과를 결론 중심으로 보여준다. 핵심은 high eta가 좁은 ridge인지, supported tube/cloud인지, 또는 lambda=35 straight path가 대표 단면이 아닌지를 구분하는 것이다.</p>
    </div>
    <nav>
      <a href="55_priorityD_robustness_dashboard.html">55 robustness</a>
      <a href="54_priority6_eigenstate_dashboard.html">54 eigenstate</a>
      <a href="50_grouped_bridge_validity_dashboard.html">50 bridge validity</a>
    </nav>
  </div>
</header>
<main>
  <section class="summary-grid" id="summaryCards"></section>
  <section class="note"><strong>핵심 메시지</strong><br><span id="keyMessage"></span></section>
  <section class="claim-grid" id="claimCards"></section>
  <section class="controls">
    <label>Geometry label <select id="labelSelect"></select></label>
    <label>Pair <select id="pairSelect"></select></label>
  </section>
  <section class="grid">
    <article class="panel"><h2>1. Radius별 eta/support 전체 경향</h2><div class="chart" id="radiusChart"></div><div class="legend"><span><i class="swatch" style="background:#245b9f"></i>eta median</span><span><i class="swatch" style="background:#14745f"></i>support low-or-better</span><span><i class="swatch" style="background:#b25b13"></i>support high/medium</span></div><p class="caption">Radius 0.25에서도 완전한 off-support는 아니지만 high/medium support는 낮아진다.</p></article>
    <article class="panel"><h2>2. Pair-level geometry label</h2><div class="chart" id="labelBar"></div><p class="caption">가장 큰 그룹은 path-not-centered 또는 lambda=35 mismatch 후보이다.</p></article>
    <article class="panel"><h2>3. Label x dominant best lambda</h2><div class="chart" id="lambdaHeat"></div><p class="caption">path/lambda mismatch 그룹에 dominant best lambda=35 pair가 없다는 점이 핵심이다.</p></article>
    <article class="panel"><h2>4. Baseline high fraction vs normal gain</h2><div class="chart" id="scatter"></div><p class="caption">왼쪽 위/오른쪽 위 후보는 normal perturbation에서 baseline보다 자주 좋아진다.</p></article>
    <article class="panel"><h2>5. 선택 pair radius profile</h2><div class="chart" id="pairRadius"></div><p class="caption">선택한 pair의 radius별 eta median과 support를 보여준다.</p></article>
    <article class="panel"><h2>6. 선택 pair 해석 카드</h2><div id="pairCard"></div></article>
    <article class="panel span2"><h2>7. High-baseline pair-alpha label</h2><div class="chart" id="alphaLabelBar"></div><p class="caption">lambda=35에서 실제 high eta인 alpha point만 보면 broad/tube 후보가 지배적이다.</p></article>
    <article class="panel span2"><h2>8. Pair table</h2><div class="table-wrap" id="pairTable"></div></article>
    <article class="panel span2"><h2>9. 다음 분석 액션</h2><div class="table-wrap" id="nextActionTable"></div></article>
  </section>
</main>
<script id="payload" type="application/json">{data}</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const COLORS = ['#245b9f','#14745f','#b25b13','#6d4cc2','#a33a48','#526071','#0f6d7a'];
const LABEL_COLORS = {{
  'path_not_centered_or_lambda35_mismatch_candidate':'#b25b13',
  'supported_high_eta_tube_or_cloud_candidate':'#14745f',
  'supported_tube_candidate':'#245b9f',
  'lambda35_low_but_retention_supported':'#6d4cc2'
}};
const LABEL_SHORT = {{
  'path_not_centered_or_lambda35_mismatch_candidate':'path/lambda mismatch',
  'supported_high_eta_tube_or_cloud_candidate':'supported tube/cloud',
  'supported_tube_candidate':'supported tube',
  'lambda35_low_but_retention_supported':'lambda35 low, retained'
}};
let state = {{label:'all', pairId:DATA.pair[0]?.pair_id || ''}};
function num(v){{const n=Number(v); return Number.isFinite(n)?n:null;}}
function fmt(v,d=3){{const n=num(v); return n===null?'not available':n.toFixed(d);}}
function pct(v){{const n=num(v); return n===null?'not available':Math.round(n*100)+'%';}}
function esc(s){{return String(s ?? 'not available').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));}}
function labelName(s){{return LABEL_SHORT[s] || s;}}
function filteredPairs(){{return DATA.pair.filter(d=>state.label==='all' || d.pair_geometry_label===state.label);}}
function currentPair(){{return DATA.pair.find(d=>d.pair_id===state.pairId) || filteredPairs()[0] || DATA.pair[0];}}
function renderSummary(){{
  const m=DATA.meta;
  const cards=[
    ['pairs',m.pair_count,'high-high core'],
    ['simulations',m.simulation_count,'solver failures 0'],
    ['lambda',m.lambda,'fixed structural slice'],
    ['radius grid',m.radius_count,'0 to 0.25'],
    ['pair-alpha',m.alpha_point_count,'sampled path points']
  ];
  document.getElementById('summaryCards').innerHTML=cards.map(c=>`<div class="metric"><b>${{esc(c[0])}}</b><span>${{esc(c[1])}}</span><small>${{esc(c[2])}}</small></div>`).join('');
  document.getElementById('keyMessage').textContent=m.key_message;
}}
function renderClaims(){{
  const keep=['C2','C3','C4','C6'];
  const claims=DATA.claims.filter(d=>keep.includes(d.claim_id));
  document.getElementById('claimCards').innerHTML=claims.map(d=>`<div class="claim"><b>${{esc(d.claim_id)}}. ${{esc(d.claim)}}</b><small>${{esc(d.strength)}}</small><p>${{esc(d.evidence)}}</p></div>`).join('');
}}
function setControls(){{
  const labels=['all',...DATA.labels.map(d=>d.pair_geometry_label)];
  document.getElementById('labelSelect').innerHTML=labels.map(l=>`<option value="${{esc(l)}}"${{state.label===l?' selected':''}}>${{esc(l==='all'?'All labels':labelName(l))}}</option>`).join('');
  const pairs=filteredPairs();
  if(!pairs.find(d=>d.pair_id===state.pairId)) state.pairId=pairs[0]?.pair_id || '';
  document.getElementById('pairSelect').innerHTML=pairs.map(d=>`<option value="${{esc(d.pair_id)}}"${{d.pair_id===state.pairId?' selected':''}}>${{esc(d.pair_id)}}</option>`).join('');
  document.getElementById('labelSelect').onchange=e=>{{state.label=e.target.value; setControls(); renderDynamic();}};
  document.getElementById('pairSelect').onchange=e=>{{state.pairId=e.target.value; renderDynamic();}};
}}
function svg(w,h,body){{return `<svg viewBox="0 0 ${{w}} ${{h}}" role="img">${{body}}</svg>`;}}
function scale(v,min,max,a,b){{if(max===min)return(a+b)/2; return a+(v-min)*(b-a)/(max-min);}}
function axes(w,h,p,yLabel,xLabel){{let g=`<line class="axis" x1="${{p.l}}" y1="${{h-p.b}}" x2="${{w-p.r}}" y2="${{h-p.b}}"/><line class="axis" x1="${{p.l}}" y1="${{p.t}}" x2="${{p.l}}" y2="${{h-p.b}}"/>`;
 for(let i=0;i<=4;i++){{const y=scale(i,0,4,h-p.b,p.t); g+=`<line class="gridline" x1="${{p.l}}" y1="${{y}}" x2="${{w-p.r}}" y2="${{y}}"/><text class="tick" x="${{p.l-8}}" y="${{y+4}}" text-anchor="end">${{(i/4).toFixed(2)}}</text>`;}}
 return g+`<text class="label" x="${{p.l}}" y="${{p.t-10}}">${{esc(yLabel)}}</text><text class="label" x="${{w-p.r}}" y="${{h-8}}" text-anchor="end">${{esc(xLabel)}}</text>`;}}
function linePath(data,xF,yF){{return data.map((d,i)=>`${{i?'L':'M'}}${{xF(d).toFixed(1)}},${{yF(d).toFixed(1)}}`).join(' ');}}
function renderRadiusChart(){{
 const w=760,h=390,p={{l:54,r:24,t:34,b:48}}, data=DATA.radius;
 let body=axes(w,h,p,'fraction / eta','radius');
 const x=d=>scale(num(d.radius),0,0.25,p.l,w-p.r), y=v=>scale(v,0,1,h-p.b,p.t);
 const keys=[['eta20_median','#245b9f'],['support_low_or_better_fraction','#14745f'],['support_high_medium_fraction','#b25b13']];
 keys.forEach(([k,c])=>{{body+=`<path d="${{linePath(data,x,d=>y(num(d[k])) )}}" fill="none" stroke="${{c}}" stroke-width="3"/>`; data.forEach(d=>body+=`<circle cx="${{x(d)}}" cy="${{y(num(d[k]))}}" r="3.5" fill="${{c}}"/>`);}});
 data.forEach(d=>body+=`<text class="tick" x="${{x(d)}}" y="${{h-p.b+17}}" text-anchor="middle">${{num(d.radius).toFixed(2).replace('0.','.') }}</text>`);
 document.getElementById('radiusChart').innerHTML=svg(w,h,body);
}}
function renderLabelBar(){{
 const w=760,h=390,p={{l:250,r:28,t:30,b:38}}, data=DATA.labels;
 const max=Math.max(...data.map(d=>num(d.n_pairs)||0));
 let body='';
 data.forEach((d,i)=>{{const y=p.t+i*72; const bw=scale(num(d.n_pairs),0,max,0,w-p.l-p.r); const c=LABEL_COLORS[d.pair_geometry_label]||COLORS[i%COLORS.length]; body+=`<text class="tick" x="${{p.l-8}}" y="${{y+26}}" text-anchor="end">${{esc(labelName(d.pair_geometry_label))}}</text><rect x="${{p.l}}" y="${{y}}" width="${{bw}}" height="38" rx="4" fill="${{c}}"/><text class="label" x="${{p.l+bw+8}}" y="${{y+25}}">${{d.n_pairs}}</text>`;}});
 body+=`<line class="axis" x1="${{p.l}}" y1="${{h-p.b}}" x2="${{w-p.r}}" y2="${{h-p.b}}"/>`;
 document.getElementById('labelBar').innerHTML=svg(w,h,body);
}}
function renderLambdaHeat(){{
 const data=DATA.label_lambda, lambdas=Object.keys(data[0]).filter(k=>k!=='pair_geometry_label');
 const w=760,h=370,p={{l:250,r:24,t:36,b:52}}, cellW=(w-p.l-p.r)/lambdas.length, cellH=52;
 const max=Math.max(...data.flatMap(r=>lambdas.map(k=>num(r[k])||0)));
 let body='';
 lambdas.forEach((l,j)=>body+=`<text class="tick" x="${{p.l+j*cellW+cellW/2}}" y="${{p.t-12}}" text-anchor="middle">${{esc(l.replace('.0',''))}}</text>`);
 data.forEach((r,i)=>{{const y=p.t+i*cellH; body+=`<text class="tick" x="${{p.l-8}}" y="${{y+31}}" text-anchor="end">${{esc(labelName(r.pair_geometry_label))}}</text>`; lambdas.forEach((l,j)=>{{const v=num(r[l])||0; const intensity=max? v/max:0; const fill=`rgba(36,91,159,${{0.10+0.85*intensity}})`; body+=`<rect x="${{p.l+j*cellW}}" y="${{y}}" width="${{cellW-3}}" height="${{cellH-5}}" rx="4" fill="${{fill}}"/><text class="label" x="${{p.l+j*cellW+cellW/2}}" y="${{y+30}}" text-anchor="middle">${{v}}</text>`;}});}});
 document.getElementById('lambdaHeat').innerHTML=svg(w,h,body);
}}
function renderScatter(){{
 const w=760,h=400,p={{l:56,r:22,t:32,b:52}}, data=DATA.pair;
 let body=axes(w,h,p,'normal gain max','baseline high alpha fraction');
 const x=d=>scale(num(d.baseline_high_alpha_fraction)||0,0,1,p.l,w-p.r), y=d=>scale(num(d.normal_gain_fraction_max)||0,0,1,h-p.b,p.t);
 data.forEach(d=>{{const c=LABEL_COLORS[d.pair_geometry_label]||'#526071'; body+=`<circle cx="${{x(d)}}" cy="${{y(d)}}" r="5" fill="${{c}}" opacity="0.78"><title>${{esc(d.pair_id)}}\\n${{esc(labelName(d.pair_geometry_label))}}</title></circle>`;}});
 for(let i=0;i<=4;i++){{const xv=scale(i/4,0,1,p.l,w-p.r); body+=`<text class="tick" x="${{xv}}" y="${{h-p.b+17}}" text-anchor="middle">${{(i/4).toFixed(2)}}</text>`;}}
 document.getElementById('scatter').innerHTML=svg(w,h,body);
}}
function renderPairRadius(){{
 const pair=currentPair(); const data=DATA.pair_radius.filter(d=>d.pair_id===pair.pair_id);
 const w=760,h=390,p={{l:54,r:24,t:34,b:48}};
 let body=axes(w,h,p,'fraction / eta','radius');
 const x=d=>scale(num(d.radius),0,0.25,p.l,w-p.r), y=v=>scale(v,0,1,h-p.b,p.t);
 [['eta20_median','#245b9f'],['support_low_or_better_fraction','#14745f'],['support_high_medium_fraction','#b25b13']].forEach(([k,c])=>{{body+=`<path d="${{linePath(data,x,d=>y(num(d[k])) )}}" fill="none" stroke="${{c}}" stroke-width="3"/>`; data.forEach(d=>body+=`<circle cx="${{x(d)}}" cy="${{y(num(d[k]))}}" r="3.5" fill="${{c}}"/>`);}});
 data.forEach(d=>body+=`<text class="tick" x="${{x(d)}}" y="${{h-p.b+17}}" text-anchor="middle">${{num(d.radius).toFixed(2).replace('0.','.') }}</text>`);
 document.getElementById('pairRadius').innerHTML=svg(w,h,body);
}}
function renderAlphaLabelBar(){{
 const w=1000,h=300,p={{l:260,r:30,t:28,b:40}}, data=DATA.pair_alpha_label_counts;
 const max=Math.max(...data.map(d=>num(d.n_pair_alpha)||0));
 let body='';
 data.forEach((d,i)=>{{const y=p.t+i*52; const bw=scale(num(d.n_pair_alpha),0,max,0,w-p.l-p.r); body+=`<text class="tick" x="${{p.l-8}}" y="${{y+28}}" text-anchor="end">${{esc(d.pair_alpha_geometry_label)}}</text><rect x="${{p.l}}" y="${{y}}" width="${{bw}}" height="34" rx="4" fill="${{COLORS[i%COLORS.length]}}"/><text class="label" x="${{p.l+bw+8}}" y="${{y+23}}">${{d.n_pair_alpha}}</text>`;}});
 document.getElementById('alphaLabelBar').innerHTML=svg(w,h,body);
}}
function renderPairCard(){{
 const d=currentPair(); const color=LABEL_COLORS[d.pair_geometry_label]||'#526071';
 const interp=DATA.label_interpretation.find(x=>x.pair_geometry_label===d.pair_geometry_label);
 document.getElementById('pairCard').innerHTML=`<div class="pair-card"><div class="pair-title"><b>${{esc(d.pair_id)}}</b><span class="pill" style="border-color:${{color}};color:${{color}}">${{esc(labelName(d.pair_geometry_label))}}</span></div>
 <div class="kv">
 <div><b>baseline high alpha fraction</b><span>${{fmt(d.baseline_high_alpha_fraction)}}</span></div>
 <div><b>R joint high max</b><span>${{fmt(d.R_joint_abs_high_median_max)}}</span></div>
 <div><b>R retention median</b><span>${{fmt(d.R_joint_retention_median)}}</span></div>
 <div><b>normal gain max</b><span>${{fmt(d.normal_gain_fraction_max)}}</span></div>
 <div><b>dominant best lambda</b><span>${{fmt(d.dominant_best_lambda,0)}}</span></div>
 <div><b>path min / max best eta</b><span>${{fmt(d.path_min_best_eta20)}} / ${{fmt(d.path_max_best_eta20)}}</span></div>
 </div>
 <p class="caption"><strong>해석:</strong> ${{esc(interp?.interpretation || 'not available')}}</p>
 <p class="caption"><strong>Caveat:</strong> ${{esc(interp?.caveat || 'not available')}}</p></div>`;
}}
function table(rows, cols){{
 return `<table><thead><tr>${{cols.map(c=>`<th>${{esc(c.label)}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(r=>`<tr>${{cols.map(c=>`<td>${{esc(c.f?c.f(r[c.key],r):r[c.key])}}</td>`).join('')}}</tr>`).join('')}}</tbody></table>`;
}}
function renderTables(){{
 const rows=filteredPairs();
 document.getElementById('pairTable').innerHTML=table(rows,[{{key:'pair_id',label:'pair'}},{{key:'label_short',label:'label'}},{{key:'baseline_high_alpha_fraction',label:'baseline high frac',f:v=>fmt(v)}},{{key:'R_joint_abs_high_median_max',label:'R joint high max',f:v=>fmt(v)}},{{key:'R_gain_max',label:'R gain max',f:v=>fmt(v)}},{{key:'normal_gain_fraction_max',label:'gain frac max',f:v=>fmt(v)}},{{key:'dominant_best_lambda',label:'dominant lambda',f:v=>fmt(v,0)}},{{key:'path_min_best_eta20',label:'path min',f:v=>fmt(v)}}]);
 document.getElementById('nextActionTable').innerHTML=table(DATA.next_actions,[{{key:'priority',label:'priority'}},{{key:'action',label:'action'}},{{key:'why',label:'why'}},{{key:'how',label:'how'}},{{key:'expected_output',label:'expected output'}}]);
}}
function renderDynamic(){{renderPairRadius(); renderPairCard(); renderTables();}}
function renderAll(){{renderSummary(); renderClaims(); setControls(); renderRadiusChart(); renderLabelBar(); renderLambdaHeat(); renderScatter(); renderAlphaLabelBar(); renderDynamic();}}
renderAll();
</script>
</body>
</html>
"""


def write_readme(html_path: Path, copy_path: Path) -> None:
    text = f"""# High-High Normal Geometry Dashboard

Generated dashboard:

- `{html_path.relative_to(ROOT).as_posix()}`
- `{copy_path.relative_to(ROOT).as_posix()}`

Rebuild:

```powershell
C:\\Users\\User\\anaconda3\\envs\\py311-cu132\\python.exe new\\build_high_high_normal_geometry_dashboard.py
```

Inputs:

- `new/bridge_normal_geometry_high_high_results_20260614/csv`
- `new/bridge_normal_geometry_high_high_conclusions_20260614/csv`
- `outputs/bridge_normal_geometry_high_high_core_lambda35_20260614/csv/normal_robustness_detail.csv`
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTMLS_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    html_text = html(payload)
    out_html = OUT_DIR / HTML_NAME
    html_copy = HTMLS_DIR / HTML_NAME
    out_html.write_text(html_text, encoding="utf-8")
    shutil.copy2(out_html, html_copy)
    write_readme(out_html, html_copy)
    print(f"wrote {out_html}")
    print(f"wrote {html_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

