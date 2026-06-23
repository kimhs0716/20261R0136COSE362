# Final report experiment scripts

이 폴더는 최종 발표자료의 2-6쪽 진단 파트에 직접 대응되는 실험 코드를 모아 둔 것이다.

주요 목적은 다음 세 가지다.

1. 정적 Hamiltonian feature만으로는 고효율 Hamiltonian의 discrete cluster를 안정적으로 주장하기 어렵다는 점을 확인한다.
2. 62k trajectory 데이터에서 dynamic-response family/reference bin을 구성하고, high-eta 내부의 response diversity를 진단한다.
3. scalar condition 또는 low-dimensional descriptor만으로 dynamic response를 충분히 지정하기 어렵다는 점을 정량화한다.

## 구성

| folder | 역할 | 최종 발표 흐름 |
| --- | --- | --- |
| `01_dynamic_response_space_and_cluster_diagnosis/` | 62k/140k 데이터에서 dynamic phenotype, archetype, reduced-space route를 분석한다. | 3-4쪽: 472D dynamics 표현, static clustering 한계, continuous/spectrum-like 해석 |
| `02_normal_vector_local_perturbation/` | high-high bridge 또는 selected path 주변의 local perturbation/normal-vector robustness를 분석한다. | 4쪽: valley가 절대적 경계라기보다 path-dependent vulnerability 후보라는 해석 |
| `03_condition_descriptor_limit/` | condition descriptor가 D-family/dynamic response를 얼마나 설명하는지 분석하고 68.x/71.x 계열 dashboard를 만든다. | 5-6쪽: structure-dynamics decoupling, scalar condition 한계, trajectory-derived readout 필요성 |

## 주의

이 스크립트들은 최종 발표자료의 근거 실험을 보존하기 위한 코드이며, 일부 파일은 원래 실험 폴더의 `new/`, `outputs/`, `htmls/` 경로를 가정한다.
대용량 raw dataset은 GitHub에 포함하지 않았으므로, 필요한 경우 repository README의 dataset 링크에서 `merged_h27_140k.npz`와 `pilot_raw_62k_lambda35.npz`를 받아 권장 경로에 배치해야 한다.

최종 모델 학습/평가 코드는 `scripts/main_model/`에, context ablation 및 원래 C1-C3 검토 코드는 `scripts/context_ablation/`에 따로 정리되어 있다.
