# Model Structure And Result Logic

## 결론

최종 보고서에서 main inverse-design 모델은 `FLOW_HTBRANCHPINNTRAJ`로 제시하는 것이 현재 결과와 더 잘 맞습니다. 이유는 단순히 target-match 평균이 조금 높아서가 아닙니다. 이 모델은 simulator가 검증한 high-target 후보를 더 많이 만들면서, 성공한 샘플들이 하나의 dynamic cluster로 몰리는 현상을 줄였습니다.

`HTBAL_CNF_MIXPRIOR`는 버릴 모델이 아닙니다. density/support 관점에서 더 보수적인 CNF 비교 모델입니다. 그러나 우리가 내세우려는 novelty가 "데이터 density fitting"이 아니라 "inverse design 가능성과 dynamic-family collapse 완화"라면, main claim의 중심에는 FLOW가 더 직접적으로 연결됩니다.

세부 수치는 아래 CSV를 보세요.

- main selection matrix: `evidence/20260623_h27_main_model_selection_audit/csv/main_model_selection_matrix.csv`
- support-filtered yield by target: `evidence/20260623_h27_main_model_selection_audit/csv/support_filtered_yield_per_target.csv`
- raw inverse-design yield: `evidence/20260623_h27_inverse_design_yield_benchmark/csv/inverse_design_yield_per_target_core_models.csv`
- success dynamic-cluster diversity: `evidence/20260623_h27_inverse_design_yield_benchmark/csv/success_only_dynamic_cluster_core_models.csv`
- scalable reference assignment: `evidence/20260623_h27_inverse_design_yield_benchmark/csv/targetmatch_scalable_reference_core_models.csv`
- NLL context: `evidence/20260623_h27_main_model_selection_audit/csv/nll_context_from_drive_metrics.csv`

## 왜 NLL이 main 기준이 아닌가

이번 연구 질문은 "학습 데이터의 likelihood를 더 잘 맞추는가"가 아닙니다. 그 질문이라면 baseline CNF가 더 방어적입니다. 실제 NLL context에서도 `FLOW_HTBRANCHPINNTRAJ`는 density fit 기준으로 불리합니다.

하지만 inverse design은 다른 질문입니다. 우리가 원하는 것은 조건을 주었을 때 simulator가 인정하는 Hamiltonian 후보를 얻는 것입니다. 그리고 그 후보들이 같은 trajectory family로만 몰리지 않아야 합니다. 따라서 main 판단 기준은 아래 순서가 됩니다.

1. exact simulator target-match yield
2. success sample의 dynamic diversity
3. target별 floor와 failure mode
4. H-space support와 robustness caveat
5. NLL은 density-fit context로만 해석

## FLOW_HTBRANCHPINNTRAJ 구조

`FLOW_HTBRANCHPINNTRAJ`는 user-facing target condition만 직접 조건으로 받되, 내부에서는 dynamic branch와 PINN-lite trajectory surrogate를 사용합니다. 핵심은 branch를 사용자가 직접 고르는 조건으로 노출하지 않는다는 점입니다. 모델은 target condition을 만족하는 H를 생성하면서, 내부 branch/trajectory signal을 통해 generated trajectory family collapse를 줄이도록 설계되었습니다.

관련 코드:

- `scripts/train_h27_dynz_pinntraj_flow.py`

  H,t to population trajectory surrogate를 학습하고, trajectory-summary consistency를 flow 학습에 보조 신호로 넣는 계열입니다.

- `scripts/train_h27_diffusion_htbranch_pinntraj.py`

  internal dynamic branch embedding을 denoiser/generator 구조에 넣는 ablation입니다. branch는 내부 latent로만 쓰이고, user-facing condition에는 노출하지 않습니다.

- `scripts/train_h27_path_dynamic_flow.py`

  H27 path/dynamic condition 기반 flow 실험의 공통 기반입니다.

- `notebooks/h27_flow_htbranchpinntraj_colab.ipynb`

  Colab 실행 wrapper입니다. 복사본에서는 `Path.cwd()` 우선으로 root를 잡도록 수정해 두었습니다.

## HTBAL_CNF_MIXPRIOR 구조

`HTBAL_CNF_MIXPRIOR`는 exact-likelihood CNF 계열 안에서 internal dynamic-mode mixture prior와 stratified branch sampling을 넣은 모델입니다. CNF baseline보다 target-match yield를 개선하고, late_high target에서는 FLOW보다 더 높은 target-match를 보입니다.

하지만 success-only dynamic cluster 결과를 보면 fast_high와 very_fast에서 성공 샘플이 거의 한 cluster로 몰립니다. 즉 이 모델은 conservative comparison으로 중요하지만, "success diversity까지 함께 확보한 inverse design"이라는 주장에는 FLOW보다 덜 직접적입니다.

관련 코드:

- `scripts/train_h27_cnf_mode_prior.py`
- `reports/h27_final_mixprior_structure_condition_diversity_20260623_kr.md`
- `reports/h27_mixprior_model_structure_and_novelty_kr.md`

## 결과를 설득하는 논리

FLOW를 main으로 세우는 논리는 "모든 지표에서 이긴다"가 아닙니다. 오히려 반대입니다. NLL, late_high floor, H-space support tail에서는 MIXPRIOR 또는 baseline이 더 방어적입니다. 그럼에도 FLOW를 main으로 둘 수 있는 이유는 우리가 풀고 싶은 문제가 그 지표들이 아니라 inverse design이기 때문입니다.

같은 generated budget에서 FLOW는 exact-high target-match 총량을 늘렸고, 특히 fast_high와 very_fast에서 simulator-validated 성공 후보를 더 많이 냈습니다. 이 성공 후보가 단순히 같은 trajectory family를 반복한 것이 아니라는 점도 중요합니다. success dynamic-cluster entropy가 fast_high와 late_high에서 MIXPRIOR보다 높게 나왔습니다.

out-of-support H는 실패 처리할 근거가 아닙니다. exact simulator validation을 통과했다면 그 샘플은 target 조건을 만족한 것입니다. 다만 train support 밖으로 나간 성공 후보는 robustness와 physical plausibility를 추가로 확인해야 합니다. 따라서 보고서에서는 out-of-support를 "반박"이 아니라 "다음 검증 단계"로 제시해야 합니다.

## 최종 보고서 문장 초안

아래 문장을 중심으로 가져가면 됩니다.

`FLOW_HTBRANCHPINNTRAJ` is used as the main inverse-design model because it increases simulator-validated high-target yield while reducing success-mode collapse for the high-target conditions where baseline and MIXPRIOR concentrate into fewer dynamic clusters. This is not a likelihood claim: CNF-based models remain stronger density-fit and support-conservative comparisons. The FLOW model is therefore presented as the model that better matches the inverse-design objective, with robustness of out-of-support validated candidates left as a follow-up check.

한국어로 쓰면:

`FLOW_HTBRANCHPINNTRAJ`를 main inverse-design 모델로 둔다. 이 선택은 likelihood가 좋아서가 아니라, exact simulator가 검증한 high-target 후보를 더 많이 만들고, 성공 샘플의 dynamic cluster collapse를 줄였기 때문이다. `HTBAL_CNF_MIXPRIOR`는 density/support 관점에서 더 보수적인 CNF 비교 모델로 남긴다. FLOW의 out-of-support 성공 후보는 무효가 아니라 robustness 후속 검증 대상이다.

## 보고서에 넣을 metric 묶음

보고서 본문에는 큰 표를 모두 붙이지 말고, 핵심 판단을 3개 묶음으로 설명하세요.

첫째, inverse-design yield입니다. 같은 512 budget 조건에서 simulator-validated target-match가 얼마나 나왔는지 보여줍니다. 이 표는 main claim의 입구입니다.

둘째, success diversity입니다. 성공한 샘플들이 reference dynamic family 또는 generated dynamic cluster에 어떻게 분포하는지 봅니다. 이 표가 novelty claim의 중심입니다.

셋째, caveat table입니다. NLL, late_high floor, H-space support tail을 넣어 과장을 피합니다. 이 표는 FLOW 선택을 약화시키는 것이 아니라, claim boundary를 명확하게 해줍니다.

## 추가로 돌리면 좋은 검증

후속 실험은 FLOW를 더 밀어주기 위한 실험이 아니라, FLOW main claim의 약한 부분을 점검하는 실험이어야 합니다.

- out-of-support target-match 후보의 lambda/noise robustness
- top-K candidate 재검증
- late_high 실패 원인 분석
- FLOW success samples의 representative trajectory visualization
- support-in vs support-out 성공 후보의 physical feature 비교
