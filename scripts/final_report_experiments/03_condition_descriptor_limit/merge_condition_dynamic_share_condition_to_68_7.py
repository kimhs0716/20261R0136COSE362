from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT / "new" / "CONDITION_DYNAMIC_DASHBOARD_SHARE_20260620"
ASSETS = SHARE / "68.7_condition_dynamic_claim_integrated_assets"
HTML_OUT = SHARE / "68.7_condition_dynamic_claim_integrated_dashboard.html"
MD_OUT = SHARE / "01_condition_claim_68p7_KR.md"
README = SHARE / "README.md"


SRC68 = ROOT / "htmls" / "68_condition_claim_storyline_core_h_per_condition_like66_assets"
SRC685 = ROOT / "htmls" / "68.5_condition_claim_supplement_dashboard_assets"


FIGS = [
    {
        "src": SRC68 / "fig00b_D_condition_median_and_betweenD_variance_expanded_conditions.png",
        "dst": "fig01_d_wise_condition_trend.png",
        "section": "1. 먼저, D별 condition 경향이 있는지 본다",
        "title": "D-wise condition trend",
        "body": "각 D group에서 condition 후보들의 robust-scaled median과 between-D variance fraction을 보여 준다.",
        "point": "일부 condition은 D별 차이를 보이지만, H structural condition의 median between-D fraction은 낮다. readout 계열은 D 차이와 더 직접적으로 맞물리는 경향도 보인다.",
    },
    {
        "src": SRC68 / "fig00_within_D_condition_spread_expanded_conditions.png",
        "dst": "fig02_within_d_condition_spread.png",
        "section": "2. 같은 D 내부의 condition variation을 본다",
        "title": "Within-D condition spread",
        "body": "같은 D group 안에서 각 condition 값이 얼마나 넓게 퍼지는지 robust-scaled IQR로 보여 준다.",
        "point": "같은 D 안에서도 H/abstract condition spread가 크게 남는다. readout은 모두 같은 양상이 아니라, early_trap_10과 residual_10은 전체적으로 spread가 낮은 편이고 source_site1_10은 D별로 더 오락가락한다. 따라서 readout 내부에서도 어떤 값은 D별로 비교적 안정적이고, 어떤 값은 group 내부 variation을 남긴다.",
    },
    {
        "src": SRC68 / "fig04_D_entropy_over_condition_bins_expanded_conditions.png",
        "dst": "fig03_d_entropy_over_condition_bins.png",
        "section": "3. 비슷한 condition 구간에 여러 D가 섞이는지 본다",
        "title": "D entropy over condition bins",
        "body": "각 condition을 quantile bin으로 나눈 뒤, bin 안에 여러 D label이 얼마나 섞이는지 entropy로 보여 준다.",
        "point": "condition bin 안에서 D entropy가 낮지 않다면, 같은 condition 범위가 여러 dynamic group을 포함한다는 뜻이다. readout bin에서는 D entropy가 더 낮아지는 경우가 있어, H/abstract와 다른 양상이 보인다.",
    },
    {
        "src": SRC685 / "fig04_like66_condition_distribution_overlap_with_h.png",
        "dst": "fig04_condition_distribution_overlap.png",
        "section": "4. feature set으로 묶어도 D 분포가 겹치는지 본다",
        "title": "Condition distribution overlap matrix",
        "body": "abstract, H structural, H+abstract, H+abstract+readout feature set에서 D group 분포가 얼마나 겹치는지 보여 준다.",
        "point": "H와 abstract condition 후보만으로는 D 분포가 충분히 분리되지 않는다. readout을 포함하면 D 분포 분리가 더 개선된다.",
    },
    {
        "src": SRC68 / "fig05_single_condition_d_classifier_accuracy_expanded_conditions.png",
        "dst": "fig05_single_condition_d_classifier_accuracy.png",
        "section": "5. condition 하나만으로 D를 맞힐 수 있는지 본다",
        "title": "Single-condition D classifier accuracy",
        "body": "각 condition 하나만 사용해 D group을 예측했을 때 balanced accuracy를 보여 준다.",
        "point": "일부 H/abstract descriptor는 신호를 갖지만, single-condition 수준에서는 dynamic group을 충분히 지정하기 어렵다. readout descriptor는 상대적으로 높은 D 구분 신호를 보이는 경우가 있다.",
    },
    {
        "src": SRC685 / "fig05_like66_condition_only_classifier_confusion_with_h.png",
        "dst": "fig06_condition_only_classifier_confusion.png",
        "section": "6. 여러 condition을 묶었을 때 어디서 좋아지는지 본다",
        "title": "Condition-only D classifier confusion",
        "body": "condition feature set별로 D group을 예측했을 때 어느 D가 맞고 어느 D가 섞이는지 보여 준다.",
        "point": "H+abstract만으로는 제한적이고, H+abstract+readout에서 성능이 크게 오른다. readout을 포함하면 어떤 D가 더 잘 분리되고 어떤 D가 여전히 섞이는지 confusion pattern으로 볼 수 있다.",
    },
    {
        "src": SRC685 / "fig06_like66_local_d_purity_with_h.png",
        "dst": "fig07_local_d_purity.png",
        "section": "7. local condition space에서도 D가 섞이는지 본다",
        "title": "Local D purity in condition space",
        "body": "condition feature space에서 가까운 이웃들이 같은 D에 속하는 비율을 보여 준다.",
        "point": "local purity가 낮거나 D별 편차가 크면, condition space 근방에서도 dynamic group이 안정적으로 분리되지 않는다는 뜻이다.",
    },
    {
        "src": SRC68 / "fig02_condition_bin_residual_diversity_expanded_conditions.png",
        "dst": "fig08_condition_bin_residual_diversity.png",
        "section": "8. condition을 맞춰도 dynamic diversity가 남는지 본다",
        "title": "Condition-bin residual diversity",
        "body": "각 condition quantile bin 안에 남는 dynamic profile spread와 eta50 IQR을 보여 준다.",
        "point": "일부 condition의 일부 bin에서는 dynamic spread나 eta IQR이 줄어드는 구간이 보인다. 따라서 condition이 전혀 정보를 주지 않는 것은 아니다. 다만 전체 bin에서 일관되게 diversity가 정리되지는 않고 spread가 남는 구간도 있어, condition 값 범위를 맞추는 것만으로 dynamic diversity를 충분히 없애기는 어렵다.",
    },
    {
        "src": SRC68 / "fig03_matched_condition_pair_distance_expanded_conditions.png",
        "dst": "fig09_matched_condition_pair_distance.png",
        "section": "9. condition이 거의 같은 pair도 dynamic이 다른지 본다",
        "title": "Matched-condition pair distance",
        "body": "같은 D 안에서 condition 값이 매우 가까운 adjacent pair를 잡고, 그 pair의 dynamic distance와 eta50 distance를 보여 준다.",
        "point": "condition이 가까워도 dynamic distance가 남는다면, condition 하나의 조절만으로 dynamic similarity를 보장하기 어렵다.",
    },
    {
        "src": SRC685 / "fig07_like66_residualized_dynamic_diversity_with_h.png",
        "dst": "fig10_residualized_dynamic_diversity.png",
        "section": "10. condition feature set으로 설명한 뒤에도 남는 dynamic variation을 본다",
        "title": "Residualized dynamic diversity",
        "body": "condition feature set으로 dynamic PC를 예측한 뒤에도 남는 residual dynamic variation을 보여 준다.",
        "point": "H+abstract feature set으로 설명력이 늘어나도 residual dynamic diversity가 남는다. readout을 넣으면 잔차가 더 줄어들어, readout이 dynamic profile variation을 더 직접적으로 설명한다는 점이 보인다.",
    },
    {
        "src": SRC685 / "fig08_like66_representative_counterexample_profiles_with_h.png",
        "dst": "fig11_condition_close_dynamic_far_counterexamples.png",
        "section": "11. 반례형 예시로 직관을 확인한다",
        "title": "H+abstract-close but dynamic-far counterexamples",
        "body": "H+abstract condition feature는 가깝지만 dynamic profile은 멀리 떨어진 대표 pair를 보여 준다.",
        "point": "population-level 통계의 직관적 예시다. 단, 대표 반례만으로 전체 결론을 일반화하지 않고 앞선 분포 통계의 보조로 읽어야 한다.",
    },
    {
        "src": SRC685 / "fig07_classifier_feature_importance.png",
        "dst": "fig12_classifier_feature_importance.png",
        "section": "12. readout 의존성을 마지막으로 점검한다",
        "title": "Top features for D-group prediction",
        "body": "condition/readout/H structural 후보 중 D-group prediction에 크게 쓰인 feature를 보여 준다.",
        "point": "source_site1_10, residual_10, early_trap_10 같은 trajectory readout이 상위에 오르면, D 재구성 성능이 H/abstract condition보다 readout 정보에 크게 의존할 수 있음을 보여 준다.",
        "caveat": "feature importance는 예측 기여도다. 인과적 mechanism 또는 제어 가능한 condition이라는 뜻은 아니다.",
    },
    {
        "src": SRC68 / "fig01_single_condition_predictive_sufficiency_expanded_conditions.png",
        "dst": "fig13_scalar_readout_sanity_check.png",
        "section": "13. 보조 sanity check: scalar output 예측",
        "title": "Single-condition predictive sufficiency",
        "body": "각 condition 하나만으로 eta50, trap20, residual20 같은 scalar dynamic readout을 얼마나 예측하는지 보여 준다.",
        "point": "이 figure는 dynamic diversity 자체를 직접 보는 핵심 근거가 아니라, 단일 condition이 기본 scalar outcome도 얼마나 설명하는지 확인하는 보조 검사다.",
    },
    {
        "src": SRC68 / "fig07_residual_eta_spread_after_one_condition_binning_expanded_conditions.png",
        "dst": "fig14_residual_eta_spread_after_one_condition_binning.png",
        "section": "14. 보조 sanity check: one-condition binning 후 eta spread",
        "title": "Residual eta spread after one-condition binning",
        "body": "각 condition 하나로 binning한 뒤 D 내부 eta50 spread가 얼마나 남는지 보여 준다.",
        "point": "eta만 놓고 보아도 condition 하나로 variation이 충분히 정리되지 않는지를 확인하는 보조 figure다.",
    },
    {
        "src": SRC68 / "fig06_single_condition_dynamic_pc_prediction_expanded_conditions.png",
        "dst": "fig15_single_condition_dynamic_pc_prediction.png",
        "section": "15. 보조 sanity check: dynamic PC 예측",
        "title": "Single-condition dynamic PC prediction",
        "body": "각 condition 하나로 dynamic profile의 저차원 요약인 dynamic PC를 얼마나 예측하는지 보여 준다.",
        "point": "condition 하나가 multi-dimensional dynamic profile의 주요 축을 어느 정도 설명하는지 보는 보조 지표다. dynamic PC는 density matrix 전체가 아니라 선택된 dynamic summary의 저차원 표현이다.",
    },
    {
        "src": SRC68 / "fig08_descriptor_eta_relation_by_D_expanded_conditions.png",
        "dst": "fig16_descriptor_eta_relation_by_d.png",
        "section": "16. 보조 sanity check: D별 condition-eta 관계",
        "title": "Descriptor eta50 relation by D",
        "body": "각 D 내부에서 condition과 eta50의 Spearman 관계 방향과 크기를 보여 준다.",
        "point": "condition-eta 관계가 D마다 달라지면, condition 효과가 단순한 전역 knob라기보다 context-dependent할 가능성이 있다.",
    },
]


STYLE = """
  <style>
    :root { --ink:#172033; --muted:#596273; --line:#d9dee8; --bg:#f7f8fb; --panel:#ffffff; --blue:#4c78a8; --orange:#f58518; }
    body { margin:0; font-family: Arial, 'Malgun Gothic', sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:30px 38px 22px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 10px; font-size:28px; }
    h2 { margin:0 0 8px; font-size:20px; }
    h3 { margin:0 0 6px; font-size:15px; color:#334155; }
    p { line-height:1.55; color:var(--muted); }
    main { max-width:1500px; margin:0 auto; padding:24px 24px 42px; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:10px; max-width:1260px; margin-top:16px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:#fbfcff; }
    .metric strong { display:block; font-size:20px; color:var(--ink); margin-bottom:3px; }
    .panel, .figure { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px 20px 22px; margin:18px 0; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }
    .step { color:var(--blue); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:5px; }
    img { display:block; max-width:100%; height:auto; margin:12px auto 8px; border:1px solid #edf0f5; border-radius:6px; background:#fff; }
    .note { background:#f3f6fb; border-left:4px solid var(--blue); padding:10px 12px; color:#394255; }
    .caveat { background:#fff8ed; border-left:4px solid var(--orange); padding:10px 12px; color:#5b4630; }
    .small { font-size:13px; }
  </style>
"""


def reset_assets() -> None:
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for fig in FIGS:
        if not fig["src"].exists():
            raise FileNotFoundError(fig["src"])
        shutil.copy2(fig["src"], ASSETS / fig["dst"])


def figure_card(i: int, fig: dict[str, str]) -> str:
    caveat = f'<p class="caveat"><b>주의.</b> {fig["caveat"]}</p>' if fig.get("caveat") else ""
    return f"""
    <section class="figure">
      <div class="step">{fig['section']}</div>
      <h2>{i}. {fig['title']}</h2>
      <p><b>무엇을 보여주나.</b> {fig['body']}</p>
      <img src="{ASSETS.name}/{fig['dst']}" alt="{fig['title']}">
      <p class="note"><b>claim과의 연결.</b> {fig['point']}</p>
      {caveat}
    </section>
"""


def write_html() -> None:
    cards = "\n".join(figure_card(i, fig) for i, fig in enumerate(FIGS, start=1))
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>68.7 Condition-Dynamic Claim Integrated Dashboard</title>
{STYLE}
</head>
<body>
  <header>
    <h1>68.7 Condition-Dynamic Claim Integrated Dashboard</h1>
    <p>이 페이지는 기존 68과 68.5를 하나로 병합한 공유용 dashboard다. 큰 질문은 하나다. <b>dynamic diversity를 모델이 골고루 학습해야 한다면, C-l1, IPR, purity, H-summary 같은 condition 값을 조절하는 것만으로 그 다양성을 충분히 표현할 수 있는가?</b> 68.7은 이 질문에 맞춰 개별 condition 진단과 feature-set 보조 진단을 한 흐름으로 재배열했다.</p>
    <div class="metrics">
      <div class="metric"><strong>22,715</strong>eta50 >= 0.80 high-eta samples</div>
      <div class="metric"><strong>10</strong>whole-D groups</div>
      <div class="metric"><strong>19</strong>condition descriptors</div>
      <div class="metric"><strong>0.373</strong>H+abstract D prediction BA</div>
      <div class="metric"><strong>0.626</strong>H+abstract+readout BA</div>
      <div class="metric"><strong>0.773</strong>H+abstract+readout dynamic PC R2</div>
    </div>
  </header>
  <main>
    <section class="panel">
      <h2>전체 연구 흐름에서의 위치</h2>
      <p>현재 방향은 확실히 분리된 discrete cluster를 찾았다는 주장보다 낮은 claim에서 출발한다. 기존 분석에서는 뚜렷하게 분리되는 cluster라기보다 연속적인 spectrum에 가까운 그림이 보였고, KNN-graph 기준으로 나눈 dynamic group 안에서는 서로 다른 dynamic 특성이 관찰되었다. 따라서 다음 질문은 `이런 dynamic diversity를 모델이 골고루 학습하게 하려면 무엇을 condition 또는 학습 신호로 줄 수 있는가`다.</p>
      <div class="grid">
        <p><b>가능한 경우.</b> condition 값만으로 dynamic group 차이와 내부 variation을 충분히 설명한다면, condition 조절을 통해 diversity를 학습시키는 방향으로 갈 수 있다. 이 경우 condition 간 correlation과 likelihood를 함께 확인해야 한다.</p>
        <p><b>부족한 경우.</b> condition 값만으로는 dynamic diversity가 충분히 정리되지 않는다면, condition knob 외에 diversity-aware representation, diversity loss/constraint, surrogate/PINN 기반 신호 같은 별도 방법이 필요하다.</p>
      </div>
    </section>
    <section class="panel">
      <h2>읽는 흐름</h2>
      <p>68.7이 실제로 확인하는 것은 두 가지다. 첫째, 같은 dynamic group 안에서 condition 값이 일정한지, 아니면 spread가 크거나 경향이 약한지 본다. 둘째, 다른 dynamic group들이 서로 다른 condition 경향을 갖는지, 아니면 비슷한 condition 범위에 섞이는지 본다. 이 두 질문 모두 condition만으로 dynamic diversity를 표현할 수 있는지 판단하기 위한 것이다.</p>
      <div class="grid">
        <p><b>1-2.</b> D별 condition 경향과 같은 D 내부 spread를 본다.</p>
        <p><b>3-7.</b> condition bin/space/classifier 기준으로 D가 충분히 분리되는지 본다.</p>
        <p><b>8-11.</b> condition 값을 맞춰도 dynamic diversity가 남는지 본다.</p>
        <p><b>12-16.</b> readout에서 따로 보이는 특징과 scalar-output 보조 검사를 확인한다.</p>
      </div>
      <p class="note"><b>용어 구분.</b> <b>H</b>는 Hamiltonian에서 계산한 구조/eigenvalue/coupling summary이고, <b>abstract</b>는 cl1/IPR/purity처럼 dynamic 상태를 압축한 scalar descriptor이며, <b>readout</b>은 early_trap_10, source_site1_10, residual_10처럼 trajectory 결과에서 직접 나온 요약값이다.</p>
      <p class="caveat"><b>readout 해석.</b> readout을 포함했을 때 D를 더 잘 구분하거나 residual dynamic diversity가 더 줄어드는 것은 중요한 관찰이다. 다만 D family를 만들 때 사용한 척도 중 trajectory가 포함되어 있었으므로, 이 결과는 순환 오류 가능성과 trajectory 정보가 D grouping에 크게 영향을 주었을 가능성을 함께 고려해야 한다. 그래서 71.7에서 이를 별도로 분석했다.</p>
    </section>
{cards}
  </main>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")


def write_md() -> None:
    md = """# 68.7 설명: condition 값만으로 dynamic diversity를 표현할 수 있는가

## 핵심 질문

68.7은 기존 68과 68.5를 합친 공유용 통합 자료다. 중심 질문은 다음이다.

> dynamic diversity를 모델이 골고루 학습해야 한다면, C-l1, IPR, purity, H-summary 같은 condition 값을 조절하는 것만으로 그 다양성을 충분히 표현할 수 있는가?

## 전체 연구 흐름

이 자료는 단독으로 갑자기 나온 condition 분석이 아니라, 다음 흐름 위에 놓인다.

1. 처음에는 확실히 구분되는 discrete dynamic cluster가 있는지 보려 했다.
2. 현재까지의 결과는 뚜렷하게 분리된 cluster라기보다 연속적인 spectrum에 가까웠다.
3. 다만 KNN-graph 기준으로 나눈 dynamic group들은 서로 다른 dynamic 특성을 보였다. 즉 연속적이더라도 그 안에서 dynamic 특성이 다르게 나타나는 구간이나 방향은 있다.
4. 따라서 dynamic distance 기준으로 group/representative를 보고, 모델이 이 diversity를 골고루 학습할 수 있어야 한다는 문제가 생긴다.
5. 그 다음 질문이 바로 `기존 condition 후보만 조절해도 이 dynamic diversity를 충분히 표현할 수 있는가`다.

68.7은 이 중 마지막 질문을 담당한다. 즉 mechanism을 증명하는 자료가 아니라, condition-only 또는 simple condition-knob 접근이 충분한지 점검하는 자료다.

## 68.7에서 실제로 확인하는 두 방향

1. 같은 dynamic group 내의 condition들이 어떤 경향을 보이는가?
   - 일정하거나 spread가 작으면 condition으로 group을 대표할 가능성이 있다.
   - std/IQR이 크거나 경향성이 약하면 condition만으로 dynamic group을 나타내기 어렵다.

2. 다른 dynamic group이 각각 다른 condition 경향성을 보이는가?
   - group마다 condition trend가 뚜렷이 다르면 condition-controlled modeling 방향을 검토할 수 있다.
   - 여러 group이 비슷한 condition 범위에 섞이면 condition만으로는 dynamic group을 구분하기 어렵고, 모델이 일부 dynamic group의 특징만 학습할 가능성이 있다.

## 결과에 따른 후속 방향

- condition 조절로 다양성을 나타낼 수 있는 경우: condition 조절을 통해 diversity를 학습시키는 방향으로 갈 수 있다. 이때 condition 간 correlation을 함께 봐야 하며, 이를 무시하고 condition을 독립적으로 설정하면 낮은 likelihood의 샘플링으로 이어질 수 있다.
- condition 조절만으로 부족한 경우: condition 말고 diversity를 학습할 수 있게 하는 방향을 찾아야 한다. 예를 들면 diversity-aware representation, diversity loss/constraint, surrogate/PINN 기반 신호 등이 후보가 된다.

이 질문에서 중요한 것은 `H`, `abstract`, `readout`을 분리해서 읽는 것이다.

- H: Hamiltonian에서 계산한 structural/eigenvalue/coupling summary. dynamics를 돌린 뒤의 결과가 아니라 H 자체의 구조 후보로 볼 수 있다.
- abstract: `cl1_10_20`, `IPR_10_20`, `purity_10_20`처럼 dynamic 상태를 압축한 scalar descriptor. raw trajectory 전체는 아니지만, H-only condition보다는 dynamics 요약에 가까운 성격을 가진다.
- readout: `early_trap_10`, `source_site1_10`, `residual_10`처럼 실제 dynamics trajectory에서 직접 나온 결과 요약. readout을 포함했을 때 D를 더 잘 구분할 수 있었지만, D family를 만들 때 사용한 척도 중 trajectory가 포함되어 있었으므로 순환 오류 가능성과 trajectory 정보가 D grouping에 크게 영향을 줄 가능성을 생각해야 한다. 이 때문에 뒤의 71.7에서 별도 분석을 진행했다.

## 분석 대상

- 기준: whole-D, `eta50 >= 0.8` high-eta subset
- 샘플 수: 22,715
- D group 수: 10
- condition descriptor 수: 19

## 68.7의 논리 순서

1. D마다 condition 경향이 다른지 본다.
2. 같은 D 내부에서도 condition 값이 넓게 퍼지는지 본다.
3. condition bin 안에 여러 D가 섞이는지 본다.
4. condition feature set 공간에서 D 분포가 겹치는지 본다.
5. condition 하나 또는 condition 묶음만으로 D를 얼마나 맞히는지 본다.
6. condition을 맞춰도 dynamic spread와 eta spread가 남는지 본다.
7. condition feature가 가까워도 dynamic profile이 멀 수 있는 반례를 확인한다.
8. readout feature가 들어갔을 때 성능이 크게 오르는지 확인하고, 이것이 71.7 순환성 점검으로 왜 이어졌는지 본다.

## 핵심 결과

- H structural condition 후보는 D 또는 eta와 일부 관련이 있지만, 단일 condition으로 dynamic group을 안정적으로 지정하기에는 약하다.
- 같은 D 내부에서도 condition spread가 크고, condition bin 안에도 D entropy와 dynamic spread가 남는다.
- H+abstract feature set으로 D를 예측하면 balanced accuracy가 0.373 수준이지만, readout까지 넣으면 0.626으로 크게 오른다.
- H+abstract+readout의 dynamic PC weighted CV R2는 0.773으로, readout이 dynamic profile reconstruction에 큰 역할을 한다.
- readout을 포함했을 때 D 구분 성능과 dynamic PC 설명력이 크게 좋아졌다.
- 따라서 현재 자료는 `condition은 무의미하다`가 아니라, `현재 condition 후보만으로는 dynamic diversity를 직접 지정하기 부족하고, readout 의존성이 크다`는 쪽을 지지한다.

## Figure별 역할

| 번호 | figure | 역할 |
| ---: | --- | --- |
| 1 | D-wise condition trend | D별 condition 중심값 차이 확인 |
| 2 | Within-D condition spread | 같은 D 내부 condition variation 확인 |
| 3 | D entropy over condition bins | 비슷한 condition 구간 안의 D mixing 확인 |
| 4 | Condition distribution overlap matrix | feature set 기준 D 분포 겹침 확인 |
| 5 | Single-condition D classifier accuracy | condition 하나만으로 D를 맞히는지 확인 |
| 6 | Condition-only D classifier confusion | condition 묶음과 readout 추가 효과 확인 |
| 7 | Local D purity in condition space | local condition neighborhood에서 D가 섞이는지 확인 |
| 8 | Condition-bin residual diversity | condition bin 안에서 줄어드는 spread와 남는 spread를 함께 확인 |
| 9 | Matched-condition pair distance | condition이 가까운 pair도 dynamic이 다른지 확인 |
| 10 | Residualized dynamic diversity | condition feature set으로 설명한 뒤 남는 dynamic variation 확인 |
| 11 | H+abstract-close but dynamic-far counterexamples | 직관적 반례 확인 |
| 12 | Top features for D-group prediction | readout 의존성 확인 |
| 13 | Single-condition predictive sufficiency | scalar readout 예측 보조 sanity check |
| 14 | Residual eta spread after one-condition binning | eta variation 보조 sanity check |
| 15 | Single-condition dynamic PC prediction | dynamic profile 저차원 요약 예측 보조 sanity check |
| 16 | Descriptor eta50 relation by D | condition-eta 관계의 D별 context dependence 확인 |

## 조심해야 할 표현

- `condition은 쓸모없다`라고 말하면 과하다.
- `H structural feature가 mechanism을 설명했다`라고 말하면 과하다.
- `readout을 넣으면 dynamic diversity를 제어할 수 있다`라고 바로 말하면 과하다.
- 더 안전한 표현은 `현재 H/abstract condition 후보만으로는 dynamic diversity를 직접 지정하기에 부족하고, readout을 포함하면 D 구분은 좋아지지만 D family 생성 척도에 trajectory가 포함되어 있었으므로 71.7에서 순환 오류 가능성과 trajectory 영향 정도를 별도로 점검했다`이다.
"""
    MD_OUT.write_text(md, encoding="utf-8")


def update_readme() -> None:
    text = README.read_text(encoding="utf-8")
    text = text.replace(
        "- `01_condition_claim_68_68p5_KR.md`\n"
        "- `02_trajectory_circularity_71p7_KR.md`\n"
        "- `68_condition_claim_storyline_core_h_per_condition_like66_dashboard.html`\n"
        "- `68.5_condition_claim_supplement_dashboard.html`\n"
        "- `71.7_trajectory_circularity_integrated_dashboard.html`\n",
        "- `01_condition_claim_68p7_KR.md`\n"
        "- `02_trajectory_circularity_71p7_KR.md`\n"
        "- `68.7_condition_dynamic_claim_integrated_dashboard.html`\n"
        "- `71.7_trajectory_circularity_integrated_dashboard.html`\n",
    )
    text = text.replace(
        "1. `01_condition_claim_68_68p5_KR.md`\n"
        "2. `68_condition_claim_storyline_core_h_per_condition_like66_dashboard.html`\n"
        "3. `68.5_condition_claim_supplement_dashboard.html`\n"
        "4. `02_trajectory_circularity_71p7_KR.md`\n"
        "5. `71.7_trajectory_circularity_integrated_dashboard.html`\n",
        "1. `01_condition_claim_68p7_KR.md`\n"
        "2. `68.7_condition_dynamic_claim_integrated_dashboard.html`\n"
        "3. `02_trajectory_circularity_71p7_KR.md`\n"
        "4. `71.7_trajectory_circularity_integrated_dashboard.html`\n",
    )
    text = text.replace(
        "68/68.5는 `condition 값을 조절하는 것만으로 dynamic group 또는 dynamic diversity를 직접 표현할 수 있는가`를 점검합니다.",
        "68.7은 기존 68/68.5를 병합한 자료로, `condition 값을 조절하는 것만으로 dynamic group 또는 dynamic diversity를 직접 표현할 수 있는가`를 점검합니다.",
    )
    README.write_text(text, encoding="utf-8")


def remove_split_files() -> None:
    targets = [
        SHARE / "01_condition_claim_68_68p5_KR.md",
        SHARE / "68_condition_claim_storyline_core_h_per_condition_like66_dashboard.html",
        SHARE / "68.5_condition_claim_supplement_dashboard.html",
        SHARE / "68_condition_claim_storyline_core_h_per_condition_like66_assets",
        SHARE / "68.5_condition_claim_supplement_dashboard_assets",
    ]
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    reset_assets()
    write_html()
    write_md()
    update_readme()
    remove_split_files()
    print(HTML_OUT)
    print(MD_OUT)


if __name__ == "__main__":
    main()
