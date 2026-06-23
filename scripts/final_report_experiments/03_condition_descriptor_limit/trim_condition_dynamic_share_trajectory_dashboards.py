from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT / "new" / "CONDITION_DYNAMIC_DASHBOARD_SHARE_20260620"


STYLE = """
  <style>
    :root { --ink:#172033; --muted:#5c6575; --line:#d9dee8; --bg:#f7f8fb; --panel:#ffffff; }
    body { margin:0; font-family: Arial, 'Malgun Gothic', sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:30px 38px 22px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 10px; font-size:28px; }
    h2 { margin:0 0 8px; font-size:20px; }
    p { line-height:1.55; color:var(--muted); }
    main { max-width:1500px; margin:0 auto; padding:24px 24px 42px; }
    .metrics { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin-top:18px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:#fbfcff; }
    .metric strong { display:block; font-size:21px; color:var(--ink); }
    .figure, .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px 20px 22px; margin:18px 0; }
    img { display:block; max-width:100%; height:auto; margin:12px auto 8px; border:1px solid #edf0f5; }
    .note { background:#f3f6fb; border-left:4px solid #4c78a8; padding:10px 12px; color:#394255; }
    .caveat { background:#fff8ed; border-left:4px solid #f58518; padding:10px 12px; color:#5b4630; }
    table { border-collapse:collapse; width:100%; font-size:13px; }
    th, td { border:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }
    th { background:#f1f4f8; }
  </style>
"""


def section(idx: int, title: str, body: str, img: str, point: str, caveat: str | None = None) -> str:
    caveat_html = f'<p class="caveat"><b>주의.</b> {caveat}</p>' if caveat else ""
    return f"""
    <section class="figure">
      <h2>{idx}. {title}</h2>
      <p>{body}</p>
      <img src="{img}" alt="{title}">
      <p class="note"><b>해석 포인트.</b> {point}</p>
      {caveat_html}
    </section>
"""


def write_html(path: str, title: str, intro: str, metrics: list[tuple[str, str]], body: str) -> None:
    metric_html = "".join(f'<div class="metric"><strong>{value}</strong>{label}</div>' for value, label in metrics)
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
{STYLE}
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>{intro}</p>
    <div class="metrics">{metric_html}</div>
  </header>
  <main>
    <p class="caveat"><b>공유용 축약 기준.</b> 이 페이지는 “trajectory가 D family 구분 결과에 얼마나 영향을 주었는가”에 직접 연결되는 figure만 남긴 축약본이다. 더 세부적인 trajectory shape 설명용 figure는 공유본에서 제외했다.</p>
{body}
  </main>
</body>
</html>
"""
    (SHARE / path).write_text(html, encoding="utf-8")


def prune_assets(asset_dir: str, keep: set[str]) -> None:
    folder = SHARE / asset_dir
    if not folder.exists():
        raise FileNotFoundError(folder)
    for p in folder.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink()


def trim_69() -> None:
    asset = "69_dynamic_group_trajectory_separation_assets"
    keep = {
        "fig01_dynamic_embedding_pca_scatter_by_D.png",
        "fig06_trajectory_metric_contribution_to_D_separation.png",
        "fig07_D_recall_from_trajectory_summary_metrics.png",
    }
    prune_assets(asset, keep)
    body = "\n".join([
        section(
            1,
            "Dynamic embedding PCA scatter",
            "D label 생성에 쓰인 full dynamic embedding을 PCA로 압축해 D별 위치를 보여 준다.",
            f"{asset}/fig01_dynamic_embedding_pca_scatter_by_D.png",
            "D family가 trajectory embedding 공간에서 실제로 분리되어 보이는지 확인하는 전체 지도다. 단, 이는 label source를 다시 보는 diagnostic이다.",
        ),
        section(
            2,
            "Trajectory metric contribution to D separation",
            "trajectory summary metric별 between-D variance fraction과 classifier importance를 비교한다.",
            f"{asset}/fig06_trajectory_metric_contribution_to_D_separation.png",
            "어떤 trajectory 요약값이 D family 구분에 크게 드러나는지 직접 보여 준다. 이 페이지에서 가장 핵심적인 figure다.",
        ),
        section(
            3,
            "D recall from trajectory summary metrics",
            "trajectory summary metric만으로 기존 D group을 재구성했을 때 group별 recall을 보여 준다.",
            f"{asset}/fig07_D_recall_from_trajectory_summary_metrics.png",
            "trajectory summary만으로도 D label을 상당 부분 맞힌다는 점을 보여 주며, D label의 trajectory 의존성을 정량적으로 보조한다.",
        ),
    ])
    write_html(
        "69_dynamic_group_trajectory_separation_dashboard.html",
        "69. Dynamic Group Trajectory Separation",
        "기존 D family가 trajectory feature 공간에서 어떻게 분리되는지, 그리고 trajectory summary가 D family 구분에 얼마나 기여하는지 보는 축약 dashboard다.",
        [("62,000", "samples"), ("13", "D groups"), ("0.862", "PCA 4D cumulative variance"), ("0.722", "summary-metric D prediction BA")],
        body,
    )


def trim_69_5() -> None:
    asset = "69.5_trajectory_grouping_supplement_dashboard_assets"
    keep = {
        "fig02_time_channel_between_D_variance_heatmap.png",
        "fig05_per_D_recall_by_trajectory_block.png",
    }
    prune_assets(asset, keep)
    body = "\n".join([
        section(
            1,
            "Time-channel D separation heatmap",
            "time x trajectory channel별 between-D variance fraction을 보여 준다.",
            f"{asset}/fig02_time_channel_between_D_variance_heatmap.png",
            "D family 구분이 특정 시간대와 trajectory channel에 강하게 드러나는지 확인한다.",
        ),
        section(
            2,
            "Per-D recall by trajectory block",
            "각 trajectory block만으로 각 D group을 얼마나 재구성할 수 있는지 보여 준다.",
            f"{asset}/fig05_per_D_recall_by_trajectory_block.png",
            "D마다 어느 trajectory block에 더 의존하는지 보여 주어, D family가 단순 eta level 이상임을 보조한다.",
            "이 결과는 supervised reconstruction이며, block별 clustering을 다시 수행한 결과는 아니다.",
        ),
    ])
    write_html(
        "69.5_trajectory_grouping_supplement_dashboard.html",
        "69.5 Trajectory Grouping Supplement",
        "69번을 보조해, D family 구분에 어느 time/channel/block 정보가 크게 드러나는지 압축해서 보여 준다.",
        [("0.921", "path-curve-only BA"), ("0.722", "summary-metric BA"), ("0.925", "early eta/trap/residual peak variance"), ("0.807", "source/site1 peak variance")],
        body,
    )


def trim_71() -> None:
    asset = "71_dynamic_group_trajectory_circularity_audit_assets"
    keep = {
        "fig01_original_dynamic_feature_inventory.png",
        "fig02_condition_vs_trajectory_prediction_ladder.png",
        "fig03_drop_one_block_performance_loss.png",
    }
    prune_assets(asset, keep)
    table = """
    <section class="panel">
      <h2>Original clustering feature audit</h2>
      <p>원래 D clustering 입력과 현재 audit feature block의 대응이다.</p>
      <table>
        <thead><tr><th>original component</th><th>current audit component</th><th>status</th><th>note</th></tr></thead>
        <tbody>
          <tr><td>eta_ds</td><td>eta_curve</td><td>matched</td><td>eta_t_dense downsampled by dynamic_downsample_stride=4.</td></tr>
          <tr><td>np.gradient(eta, times)</td><td>eta_slope</td><td>matched</td><td>same finite-difference eta slope construction.</td></tr>
          <tr><td>path_ds</td><td>all_path_curves</td><td>matched</td><td>same 7 site/path group population curves kept as one original clustering block.</td></tr>
          <tr><td>dynamic_metrics(...)</td><td>summary_metrics</td><td>matched</td><td>eta10/20/50, t80/t90, tau, residence, final path population metrics.</td></tr>
        </tbody>
      </table>
    </section>
"""
    body = table + "\n".join([
        section(
            1,
            "Original D grouping input feature inventory",
            "원래 D grouping dynamic embedding이 eta curve, eta slope, path curves, summary metrics 중 어떤 feature count 구성을 갖는지 보여 준다.",
            f"{asset}/fig01_original_dynamic_feature_inventory.png",
            "D label 자체가 trajectory curve 중심 feature에서 만들어졌다는 점을 명시한다.",
        ),
        section(
            2,
            "Condition vs trajectory circularity ladder",
            "eta50 >= 0.8 subset에서 condition 후보, readout, trajectory feature가 D group을 얼마나 예측/재구성하는지 비교한다.",
            f"{asset}/fig02_condition_vs_trajectory_prediction_ladder.png",
            "trajectory feature가 readout 제외 condition 후보보다 훨씬 강하게 D를 재구성한다. 순환 오류 논의의 핵심 figure다.",
        ),
        section(
            3,
            "Drop-one-block ablation",
            "full trajectory embedding에서 원래 D grouping input block 하나를 제거했을 때 D 재구성 balanced accuracy가 얼마나 떨어지는지 보여 준다.",
            f"{asset}/fig03_drop_one_block_performance_loss.png",
            "원래 D family 구분에서 어떤 trajectory block이 대체되기 어려운 정보를 담는지 본다. all_path_curves 제거가 가장 큰 하락을 만든다.",
            "drop-one 하락이 작다고 해당 block이 무의미하다는 뜻은 아니다. 다른 block과 정보가 중복될 수 있다.",
        ),
    ])
    write_html(
        "71_dynamic_group_trajectory_circularity_audit_dashboard.html",
        "71. Dynamic Group Trajectory Circularity Audit",
        "condition 실험에서 발견된 순환 오류 가능성을 점검하기 위해, D family의 원래 trajectory 입력과 condition/trajectory reconstruction 성능을 직접 비교한다.",
        [("62,000", "all samples"), ("13", "D groups"), ("0.928", "full trajectory BA"), ("0.386", "condition-only BA")],
        body,
    )


def trim_71_5() -> None:
    asset = "71.5_path_channel_supplement_dashboard_assets"
    keep = {
        "fig01_single_channel_D_reconstruction.png",
        "fig03_time_resolved_between_D_variance.png",
    }
    prune_assets(asset, keep)
    body = "\n".join([
        section(
            1,
            "Single-channel D reconstruction",
            "source/site1, trap, residual, site2 curve만으로 D label을 얼마나 재구성하는지 보여 준다.",
            f"{asset}/fig01_single_channel_D_reconstruction.png",
            "source/trap/residual이 D label과 얼마나 연결되어 있는지 가장 직접적으로 보여 준다.",
            "path block 내부 정보를 다시 읽는 것이므로 독립 검증은 아니다.",
        ),
        section(
            2,
            "Time-resolved source/trap/residual separation",
            "source/site1, site2, trap, residual의 시간별 between-D variance fraction을 보여 준다.",
            f"{asset}/fig03_time_resolved_between_D_variance.png",
            "D family 구분이 source/trap/residual trajectory의 어느 시간대에서 강하게 드러나는지 확인한다.",
            "이 페이지는 71번의 핵심 circularity audit를 보조하는 channel-level diagnostic이다.",
        ),
    ])
    write_html(
        "71.5_path_channel_supplement_dashboard.html",
        "71.5 Path-Channel Supplement",
        "71번의 original-block audit를 보조해, source/site1, trap, residual channel이 D family와 얼마나 연결되어 있는지 압축해서 보여 준다.",
        [("0.548", "source/site1 BA"), ("0.567", "trap BA"), ("0.567", "residual BA"), ("0.420", "site2 BA")],
        body,
    )


def update_md() -> None:
    md = """# 69-71.5 설명: D family 구분에 trajectory가 얼마나 영향을 주었는가

## 핵심 질문

69번부터 71.5번까지는 앞선 condition 실험에서 발견된 순환 오류 가능성을 점검하기 위한 자료다.

문제는 다음과 같다.

1. D family는 애초에 trajectory feature를 포함한 dynamic embedding으로 구분되었다.
2. 그런데 condition 실험에서 trajectory/readout 성격의 값을 condition처럼 포함하면, D family를 잘 맞히는 결과가 독립적인 근거처럼 보일 수 있다.
3. 따라서 여기서는 trajectory가 D family 구분 결과에 얼마나 영향을 주었는지, 그리고 condition 실험에서 어떤 부분을 순환적으로 해석하면 안 되는지를 확인한다.

## 공유용 figure 축약 기준

이 공유 폴더에서는 핵심 질문에 직접 답하는 figure만 남겼다. D별 trajectory shape를 자세히 설명하는 산점도 grid, median curve, 개별 feature ranking 등은 공유본에서 제외했다.

남긴 figure는 다음 역할을 한다.

- 69: D family가 trajectory embedding/summary metric으로 얼마나 재구성되는지 보여 주는 입구 자료.
- 69.5: 어떤 time/channel/block의 trajectory 정보가 D family 구분에 크게 드러나는지 보여 주는 보조 자료.
- 71: trajectory-derived D label을 condition/readout/trajectory로 다시 맞히는 circularity audit의 핵심 자료.
- 71.5: source/trap/residual channel이 D label과 얼마나 연결되어 있는지 보는 channel-level 보조 자료.

## 69번: D family가 trajectory space에서 분리되는가

69번은 전체 샘플 62,000개와 D group 13개를 대상으로 한다.

핵심 수치:

| 항목 | 값 |
| --- | ---: |
| dynamic PCA 4D cumulative explained variance | 0.862 |
| trajectory summary metric 13개만으로 D reconstruction BA | 0.722 |

남긴 figure:

1. `Dynamic embedding PCA scatter`
   - D family가 trajectory embedding 공간에서 실제로 분리되어 보이는지 확인한다.
2. `Trajectory metric contribution to D separation`
   - 어떤 trajectory summary metric이 D family 차이를 크게 드러내는지 보여 준다.
3. `D recall from trajectory summary metrics`
   - trajectory summary만으로도 어떤 D group이 잘 재구성되는지 확인한다.

해석:

- D family는 eta level뿐 아니라 도달 시간, loss, residual, sink/detour residence 같은 trajectory shape 정보를 담는다.
- summary metric만으로도 BA 0.722가 나오므로 D label이 trajectory profile 차이를 담고 있다는 sanity check가 된다.
- 하지만 이것은 label source를 다시 보는 것이므로 독립 검증은 아니다.

## 69.5번: 어느 trajectory time/channel/block이 D 구분에 기여하는가

69.5번은 69번의 보조 자료다.

핵심 수치:

| feature block 또는 channel | 값 |
| --- | ---: |
| path curve all channels only BA | 0.921 |
| trajectory summary metrics BA | 0.722 |
| eta/trap/residual early peak between-D variance | 약 0.925 |
| source/site1 peak between-D variance | 0.807 |

남긴 figure:

1. `Time-channel D separation heatmap`
   - 어느 시간대와 trajectory channel에서 D 차이가 크게 드러나는지 보여 준다.
2. `Per-D recall by trajectory block`
   - D group마다 어떤 trajectory block으로 재구성이 잘 되는지 보여 준다.

해석:

- D family는 단순 eta scalar만이 아니라 path population curve 전체와 시간별 route 차이를 강하게 반영한다.
- 특히 path curve all channels만으로 BA 0.921이 나온다.

## 71번: circularity audit의 핵심

71번은 이번 묶음의 핵심이다. 원래 D grouping에 실제로 들어간 block만 기준으로 audit했다.

원래 D grouping block:

| original block | current audit component | features |
| --- | --- | ---: |
| eta_ds | eta_curve | 51 |
| gradient(eta) | eta_slope | 51 |
| path_ds | all_path_curves | 357 |
| dynamic_metrics | summary_metrics | 13 |
| total | full_dynamic_embedding | 472 |

condition-vs-trajectory 비교:

| source | subset | balanced accuracy |
| --- | --- | ---: |
| trajectory full dynamic embedding | eta50 >= 0.8 | 0.910 |
| trajectory path curve all channels | eta50 >= 0.8 | 0.904 |
| trajectory summary metrics | eta50 >= 0.8 | 0.711 |
| H+abstract+readout | eta50 >= 0.8 | 0.677 |
| readout 제외 H+abstract condition | eta50 >= 0.8 | 0.386 |

drop-one-block 결과:

| removed block | balanced accuracy | delta vs full |
| --- | ---: | ---: |
| none, full embedding | 0.928 | 0.000 |
| all_path_curves 제거 | 0.772 | -0.156 |
| eta_slope 제거 | 0.922 | -0.006 |
| summary_metrics 제거 | 0.927 | -0.001 |
| eta_curve 제거 | 0.930 | +0.002 |

남긴 figure:

1. `Original D grouping input feature inventory`
   - D label 자체가 trajectory curve 중심 feature에서 만들어졌음을 보여 준다.
2. `Condition vs trajectory circularity ladder`
   - trajectory/readout이 readout 제외 condition 후보보다 D를 훨씬 잘 재구성함을 보여 준다.
3. `Drop-one-block ablation`
   - 원래 D grouping input 중 어떤 block을 제거하면 D 재구성이 약해지는지 보여 준다.

해석:

- high-eta subset에서도 trajectory full embedding은 BA 0.910, readout 제외 H+abstract condition은 BA 0.386이다.
- 따라서 trajectory/readout으로 D를 잘 맞히는 결과는 독립 검증이 아니라 label-source reconstruction으로 해석해야 한다.
- all_path_curves 제거가 가장 큰 하락을 만들기 때문에, D family는 path population curve 정보를 강하게 반영한다.

## 71.5번: source/trap/residual channel-level 보조 자료

71.5번은 71번 본문 근거가 아니라 channel-level 보조 diagnostic이다.

single-channel D reconstruction:

| channel | balanced accuracy |
| --- | ---: |
| source/site1 | 0.548 |
| trap | 0.567 |
| residual | 0.567 |
| source-side/site2 | 0.420 |

peak time-resolved D separation:

| channel | peak time | between-D variance fraction |
| --- | ---: | ---: |
| trap | 5.0 ps | 0.925 |
| residual | 5.25 ps | 0.925 |
| source/site1 | 20.5 ps | 0.807 |
| source-side/site2 | 26.5 ps | 0.651 |

남긴 figure:

1. `Single-channel D reconstruction`
   - source/trap/residual curve만으로 D label이 얼마나 재구성되는지 보여 준다.
2. `Time-resolved source/trap/residual separation`
   - source/trap/residual 차이가 어느 시간대에 D family 차이로 드러나는지 보여 준다.

## 현재 말할 수 있는 결론

1. D family는 trajectory-derived dynamic label이다.
2. trajectory/readout으로 D를 잘 맞히는 것은 예상되는 결과이며, condition 실험의 독립 근거로 쓰면 순환적이다.
3. high-eta subset에서 trajectory full embedding은 D를 매우 잘 재구성하지만, readout 제외 condition 후보는 훨씬 약하다.
4. 따라서 68/68.5의 condition claim을 말할 때는 trajectory/readout을 condition 후보와 분리해서 해석해야 한다.

## 조심해야 할 표현

- `trajectory로 D를 잘 맞혔으니 D group이 독립 검증되었다`라고 말하면 안 된다.
- `source/trap/residual이 D를 만든다`라고 말하면 안 된다.
- `condition 후보가 낮은 성능이므로 dynamic diversity가 없다`라고 말하면 안 된다.
- 안전한 표현은 `D label은 trajectory-derived dynamic phenotype이고, condition 후보만으로 그 phenotype을 충분히 재구성하기는 어렵다`이다.
"""
    (SHARE / "02_trajectory_label_audit_69_71p5_KR.md").write_text(md, encoding="utf-8")


def update_readme() -> None:
    path = SHARE / "README.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "4. `02_trajectory_label_audit_69_71p5_KR.md`\n   - 69번부터 71.5번까지의 D group trajectory provenance와 circularity audit을 정리한 설명 문서.",
        "4. `02_trajectory_label_audit_69_71p5_KR.md`\n   - 69번부터 71.5번까지의 D family trajectory 의존성과 circularity audit을 정리한 설명 문서. 공유본에서는 핵심 figure만 남겼다.",
    )
    text = text.replace(
        "6. `69.5_trajectory_grouping_supplement_dashboard.html`\n   - 69번의 보조 자료. trajectory grouping에 어떤 시간/채널/feature가 기여하는지 추가로 확인한다.",
        "6. `69.5_trajectory_grouping_supplement_dashboard.html`\n   - 69번의 보조 자료. trajectory grouping에 어떤 시간/채널/block이 크게 드러나는지 핵심 figure만 확인한다.",
    )
    text = text.replace(
        "8. `71.5_path_channel_supplement_dashboard.html`\n   - 71번의 보조 자료. source/site1, trap, residual 등 path-channel 관점에서 D group과 trajectory readout의 관계를 추가로 보여 준다.",
        "8. `71.5_path_channel_supplement_dashboard.html`\n   - 71번의 보조 자료. source/site1, trap, residual channel과 D group의 관계만 압축해서 보여 준다.",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    trim_69()
    trim_69_5()
    trim_71()
    trim_71_5()
    update_md()
    update_readme()
    print("trimmed trajectory share dashboards")


if __name__ == "__main__":
    main()
