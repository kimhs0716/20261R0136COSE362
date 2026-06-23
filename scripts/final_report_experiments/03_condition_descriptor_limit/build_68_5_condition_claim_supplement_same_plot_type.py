from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "new" / "whole_d_condition_claim_storyline_h_augmented_like66_20260620"
SOURCE_FIG = SOURCE / "figures"
SOURCE_CSV = SOURCE / "csv"
SOURCE_METRICS = SOURCE / "metrics_summary.json"
H_STRUCTURAL_STORYLINE = ROOT / "new" / "whole_d_condition_claim_storyline_h_structural_20260620"
H_STRUCTURAL_FIG = H_STRUCTURAL_STORYLINE / "figures"

OUT = ROOT / "new" / "condition_claim_supplement_expanded_20260620"
REPORT = OUT / "reports"
FIG = OUT / "figures"
HTMLS = ROOT / "htmls"
ASSETS = HTMLS / "68.5_condition_claim_supplement_dashboard_assets"
HTML_OUT = HTMLS / "68.5_condition_claim_supplement_dashboard.html"

FEATURE_SET_ORDER = ["abstract", "H_structural", "H_plus_abstract", "H_plus_abstract_plus_readout"]
FEATURE_SET_LABELS = {
    "abstract": "abstract summaries",
    "H_structural": "H structural",
    "H_plus_abstract": "H + abstract",
    "H_plus_abstract_plus_readout": "H + abstract + readout",
}
CONFUSION_FILES = {
    "abstract": "confusion_abstract.csv",
    "H_structural": "confusion_H_structural.csv",
    "H_plus_abstract": "confusion_H_plus_abstract.csv",
    "H_plus_abstract_plus_readout": "confusion_H_plus_abstract_plus_readout.csv",
}


FIGURES = [
    {
        "file": "fig04_like66_condition_distribution_overlap_with_h.png",
        "title": "Condition distribution overlap matrix",
        "shows": "서로 다른 D group의 condition 분포가 feature space 안에서 얼마나 겹치는지 보여준다.",
        "role": "다른 D가 서로 다른 condition 경향을 갖는지 확인하는 보조 그림이다. H structural, H+abstract, H+abstract+readout feature set까지 같은 plot 유형으로 비교한다.",
        "caveat": "overlap은 centroid-radius 기반 proxy이므로, D 분리가 완전히 된다/안 된다는 최종 판정이 아니라 분포 겹침 정도를 보는 진단으로 읽어야 한다.",
    },
    {
        "file": "fig05_like66_condition_only_classifier_confusion_with_h.png",
        "title": "Condition-only D classifier confusion",
        "shows": "condition feature만으로 D group을 예측했을 때 어느 D가 맞고 어느 D가 서로 섞이는지 보여준다.",
        "role": "condition 후보가 dynamic group을 직접 구분할 만큼 충분한지 보는 가장 직접적인 보조 그림이다. H feature를 넣었을 때와 readout까지 넣었을 때의 차이를 분리해서 볼 수 있다.",
        "caveat": "D label 자체가 trajectory-derived이므로, 이것은 mechanism validation이 아니라 condition/readout이 D label을 얼마나 재구성하는지 보는 audit이다.",
    },
    {
        "file": "fig06_like66_local_d_purity_with_h.png",
        "title": "Local D purity in condition space",
        "shows": "condition feature space에서 가까운 이웃들이 같은 D에 속하는 비율을 D별로 보여준다.",
        "role": "전역 classifier가 놓칠 수 있는 local separability를 확인한다. 같은 condition 근방에 여러 D가 섞이면 condition만으로 dynamic diversity를 안정적으로 지정하기 어렵다는 근거가 된다.",
        "caveat": "local purity가 높아도 해당 feature가 dynamics를 인과적으로 만든다는 뜻은 아니다.",
    },
    {
        "file": "fig07_like66_residualized_dynamic_diversity_with_h.png",
        "title": "Residualized dynamic diversity",
        "shows": "condition feature로 dynamic PC를 예측한 뒤에도 남는 dynamic variation의 크기를 보여준다.",
        "role": "condition 후보가 dynamic profile의 어느 정도를 설명하는지, 그리고 H structural/readout/D label을 추가했을 때 잔차가 얼마나 줄어드는지 확인한다.",
        "caveat": "dynamic PC는 trajectory profile의 저차원 요약이다. density matrix 전체를 설명했다는 의미로 읽으면 안 된다.",
    },
    {
        "file": "fig08_like66_representative_counterexample_profiles_with_h.png",
        "title": "H+abstract-close but dynamic-far counterexamples",
        "shows": "H+abstract condition feature는 가깝지만 dynamic profile은 멀리 떨어진 대표 pair를 직접 비교한다.",
        "role": "condition 값을 꽤 맞춰도 dynamic 결과가 달라질 수 있음을 직관적으로 보여 주는 반례형 보조 그림이다.",
        "caveat": "대표 반례는 population-level 통계의 예시다. 이 그림만으로 전체 분포를 일반화하지 않는다.",
    },
    {
        "file": "fig07_classifier_feature_importance.png",
        "source_dir": H_STRUCTURAL_FIG,
        "title": "Top features for D-group prediction",
        "shows": "condition/readout/H-structural 후보 중 어떤 feature가 D-group prediction에 가장 크게 쓰였는지 보여준다.",
        "role": "68.5의 confusion과 residualized dynamic diversity 결과에서 readout을 추가했을 때 성능이 크게 오르는 이유를 해석하는 보조 figure다. 특히 source_site1_10, residual_10, early_trap_10 같은 trajectory readout이 상위에 올라와, D 재구성 성능이 순수 condition 후보보다 readout 정보에 강하게 의존할 수 있음을 보여 준다.",
        "caveat": "feature importance는 classifier 내부의 예측 기여도다. 이 값이 dynamic diversity의 원인이나 제어 가능한 condition이라는 뜻은 아니다.",
    },
]


def require_inputs() -> None:
    missing = [str(SOURCE_METRICS)]
    missing.extend(str(item.get("source_dir", SOURCE_FIG) / item["file"]) for item in FIGURES)
    missing.extend(str(SOURCE_CSV / name) for name in [
        "D_classifier_feature_set_comparison_with_h.csv",
        "dynamic_pc_prediction_summary_with_h.csv",
        "condition_distribution_overlap_feature_sets_with_h.csv",
        "local_D_purity_feature_sets_with_h.csv",
        "representative_counterexample_pairs_H_plus_abstract.csv",
    ])
    missing = [path for path in missing if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))


def prepare_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    HTMLS.mkdir(parents=True, exist_ok=True)
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True, exist_ok=True)


def copy_figures() -> list[dict[str, str]]:
    copied = []
    for item in FIGURES:
        src = item.get("source_dir", SOURCE_FIG) / item["file"]
        dst = ASSETS / item["file"]
        shutil.copy2(src, dst)
        new_item = dict(item)
        new_item["src"] = f"{ASSETS.name}/{item['file']}"
        copied.append(new_item)
    return copied


def save_replotted(fig: plt.Figure, name: str) -> None:
    out = FIG / name
    asset = ASSETS / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out, asset)


def redraw_overlap_heatmap() -> None:
    source = pd.read_csv(SOURCE_CSV / "condition_distribution_overlap_feature_sets_with_h.csv")
    d_order = sorted(source["D_a"].astype(str).unique())
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 14.2), constrained_layout=True)
    im = None
    for ax, feature_set in zip(axes.ravel(), FEATURE_SET_ORDER):
        sub = source[source["feature_set"].eq(feature_set)]
        mat = (
            sub.pivot(index="D_a", columns="D_b", values="overlap")
            .reindex(index=d_order, columns=d_order)
            .to_numpy(float)
        )
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(FEATURE_SET_LABELS[feature_set], fontsize=13)
        ax.set_xticks(range(len(d_order)))
        ax.set_xticklabels(d_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(d_order)))
        ax.set_yticklabels(d_order, fontsize=8)
        ax.set_xlabel("D_b")
        ax.set_ylabel("D_a")
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.03, label="distribution overlap proxy")
    fig.suptitle("D-pair condition distribution overlap with H feature sets", fontsize=16)
    save_replotted(fig, "fig04_like66_condition_distribution_overlap_with_h.png")


def redraw_confusion_heatmap(metrics: dict[str, float]) -> None:
    d_order: list[str] | None = None
    mats: dict[str, pd.DataFrame] = {}
    for feature_set, name in CONFUSION_FILES.items():
        df = pd.read_csv(SOURCE_CSV / name, index_col=0)
        if d_order is None:
            d_order = list(df.index.astype(str))
        mats[feature_set] = df.reindex(index=d_order, columns=d_order)
    assert d_order is not None

    ba_lookup = {
        "abstract": metrics["abstract_balanced_accuracy"],
        "H_structural": metrics["h_balanced_accuracy"],
        "H_plus_abstract": metrics["h_abs_balanced_accuracy"],
        "H_plus_abstract_plus_readout": metrics["h_abs_read_balanced_accuracy"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(17.0, 14.2), constrained_layout=True)
    im = None
    for ax, feature_set in zip(axes.ravel(), FEATURE_SET_ORDER):
        mat = mats[feature_set].to_numpy(float)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{FEATURE_SET_LABELS[feature_set]}\nbalanced acc={ba_lookup[feature_set]:.3f}", fontsize=13)
        ax.set_xticks(range(len(d_order)))
        ax.set_xticklabels(d_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(d_order)))
        ax.set_yticklabels(d_order, fontsize=8)
        ax.set_xlabel("predicted D")
        ax.set_ylabel("true D")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] >= 0.18:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#172033")
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.03, label="row-normalized fraction")
    fig.suptitle("D classifier confusion from condition feature sets", fontsize=16)
    save_replotted(fig, "fig05_like66_condition_only_classifier_confusion_with_h.png")


def redraw_layout_sensitive_figures(metrics: dict[str, float]) -> None:
    redraw_overlap_heatmap()
    redraw_confusion_heatmap(metrics)


def write_html(figures: list[dict[str, str]], metrics: dict[str, float]) -> None:
    cards = []
    for i, fig in enumerate(figures, start=1):
        cards.append(
            f"""
    <section class="figure">
      <h2>{i}. {fig['title']}</h2>
      <p><b>무엇을 보여주나.</b> {fig['shows']}</p>
      <img src="{fig['src']}" alt="{fig['title']}">
      <p class="note"><b>claim과의 연결.</b> {fig['role']}</p>
      <p class="caveat"><b>주의.</b> {fig['caveat']}</p>
    </section>
"""
        )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>68.5 Condition Claim Supplement - Same Plot Types with H Conditions</title>
  <style>
    :root {{ --ink:#172033; --muted:#596273; --line:#d9dee8; --bg:#f7f8fb; --panel:#ffffff; }}
    body {{ margin:0; font-family: Arial, 'Malgun Gothic', sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:30px 38px 22px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:28px; }}
    h2 {{ margin:0 0 8px; font-size:20px; }}
    p {{ line-height:1.55; color:var(--muted); }}
    main {{ max-width:1500px; margin:0 auto; padding:24px 24px 42px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:6px 10px; background:#f7f9fc; color:#394255; font-size:13px; }}
    .figure {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px 20px 22px; margin:18px 0; }}
    img {{ display:block; max-width:100%; height:auto; margin:12px auto 8px; border:1px solid #edf0f5; }}
    .note {{ background:#f3f6fb; border-left:4px solid #4c78a8; padding:10px 12px; color:#394255; }}
    .caveat {{ background:#fff8ed; border-left:4px solid #f58518; padding:10px 12px; color:#5b4630; }}
  </style>
</head>
<body>
  <header>
    <h1>68.5 Condition Claim Supplement</h1>
    <p>기존 63번 보조 plot 유형은 유지하되, 68번에서 추가한 H structural condition 후보를 반영해 다시 계산한 보조 시각화다. 이 페이지는 condition 값만으로 dynamic diversity를 직접 지정하기 어렵다는 claim을 보조적으로 점검하기 위한 자료다.</p>
    <div class="meta">
      <span class="pill">eta50 >= 0.8 high-eta subset</span>
      <span class="pill">n = {int(metrics['n_samples']):,}</span>
      <span class="pill">D groups = {int(metrics['n_d_groups'])}</span>
      <span class="pill">abstract BA = {metrics['abstract_balanced_accuracy']:.3f}</span>
      <span class="pill">H+abstract BA = {metrics['h_abs_balanced_accuracy']:.3f}</span>
      <span class="pill">H+abstract+readout BA = {metrics['h_abs_read_balanced_accuracy']:.3f}</span>
    </div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")


def write_report(figures: list[dict[str, str]], metrics: dict[str, float]) -> None:
    lines = [
        "# 68.5 condition claim supplement 재생성 기록",
        "",
        "## 변경 내용",
        "",
        "- 이전 68.5 보조 dashboard의 plot 종류를 유지했다.",
        "- 단순 복사본이 아니라 `whole_d_condition_claim_storyline_h_augmented_like66_20260620`에서 H structural condition 후보를 포함해 다시 계산한 figure를 사용했다.",
        "- 따라서 68번에서 늘어난 condition 후보가 반영된 상태로, 기존 보조 plot의 해석 흐름을 유지한다.",
        "",
        "## 기준",
        "",
        f"- 샘플: eta50 >= 0.8 high-eta subset, n={int(metrics['n_samples']):,}",
        f"- D groups: {int(metrics['n_d_groups'])}",
        f"- abstract-only D balanced accuracy: {metrics['abstract_balanced_accuracy']:.3f}",
        f"- H structural-only D balanced accuracy: {metrics['h_balanced_accuracy']:.3f}",
        f"- H+abstract D balanced accuracy: {metrics['h_abs_balanced_accuracy']:.3f}",
        f"- H+abstract+readout D balanced accuracy: {metrics['h_abs_read_balanced_accuracy']:.3f}",
        f"- H+abstract dynamic PC weighted CV R2: {metrics['h_abs_dynamic_pc_r2']:.3f}",
        f"- H+abstract+readout dynamic PC weighted CV R2: {metrics['h_abs_read_dynamic_pc_r2']:.3f}",
        "",
        "## 포함 figure",
        "",
    ]
    for i, fig in enumerate(figures, start=1):
        lines.extend([
            f"### {i}. {fig['title']}",
            "",
            f"- 보여주는 것: {fig['shows']}",
            f"- claim과의 연결: {fig['role']}",
            f"- 주의: {fig['caveat']}",
            f"- source: `{fig['src']}`",
            "",
        ])
    (REPORT / "68_5_same_plot_type_h_condition_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    require_inputs()
    prepare_dirs()
    with SOURCE_METRICS.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    figures = copy_figures()
    redraw_layout_sensitive_figures(metrics)
    write_html(figures, metrics)
    write_report(figures, metrics)
    print(json.dumps({
        "html": str(HTML_OUT),
        "assets": str(ASSETS),
        "report": str(REPORT / "68_5_same_plot_type_h_condition_report.md"),
        "n_figures": len(figures),
        "n_samples": metrics["n_samples"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
