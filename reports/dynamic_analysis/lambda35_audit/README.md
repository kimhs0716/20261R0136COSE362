# Lambda=35 Dynamic Family Audit

이 폴더는 `lambda=35` simulator setting에서 정의한 dynamic-response family와 structural family의 보조 진단 결과를 모은 것이다.

## 포함 내용

- `figures/01_dynamic_family_profile_heatmap.png`: D-family별 dynamic response profile 요약
- `figures/02_dynamic_family_eta_tau_map.png`: D-family가 eta/tau 공간에서 어떻게 분포하는지 확인
- `figures/03_sd_association_summary.png`: structural family `S`와 dynamic family `D`의 대응 관계 요약
- `figures/04_condition_dynamic_clustering_quality.png`: condition별 dynamic clustering 품질 지표
- `tables/dynamic_family_summary_selected.csv`: D-family별 대표 scalar/dynamic summary
- `tables/sd_association_key_metrics.csv`: S-D association 지표
- `tables/legacy_condition_dynamic_clustering_summary.csv`: condition별 dynamic clustering summary
- `standard_fmo_dynamic_family_assignment.md`: 표준 FMO Hamiltonian이 어떤 dynamic family에 가까운지 확인한 보조 분석

## 해석 위치

이 자료는 D-family가 독립적인 물리 mechanism을 증명한다는 주장을 위한 것이 아니다. 또한 전역적인 physical corridor나 manifold를 증명하는 자료도 아니다. 최종 보고서에서는 다음 정도의 제한된 역할로 사용하는 것이 안전하다.

1. raw/static Hamiltonian feature만으로는 transfer behavior 차이를 충분히 요약하기 어렵다.
2. trajectory-derived D-family는 생성 모델의 dynamic-response coverage와 collapse를 진단하는 reference lens로 사용할 수 있다.
3. S-family와 D-family는 완전한 일대일 대응이 아니므로, structural label만으로 dynamic behavior를 안정적으로 지정하기는 어렵다.
4. 일부 pairwise bridge/local valley 분석은 global corridor claim이 아니라, 연결된 구조 위에서도 경로에 따라 dynamic response가 취약해질 수 있음을 보여주는 보조 diagnostic으로만 해석한다.
