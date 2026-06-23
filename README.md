# FMO Hamiltonian Conditional Flow

COSE362 기계학습 프로젝트 **Team K1** 코드 저장소입니다.

본 repository는 FMO Hamiltonian 조건부 생성 프로젝트에서 사용한 코드, 분석 스크립트, 결과 파일, 보고서 보조 자료를 정리한 것입니다.

## 주요 파일

처음 확인할 때는 아래 파일과 폴더를 우선 보면 됩니다.

- `src/`: Hamiltonian 처리, simulator wrapper, context feature, NSF 모델 등 재사용 코드
- `scripts/main_model/`: FLOW 계열 최종 모델 학습, model-selection audit, inverse-design yield benchmark 스크립트
- `scripts/context_ablation/`: context ablation, nearest-neighbor baseline, C1-C3 검토 스크립트
- `scripts/final_report_experiments/`: 최종 발표자료의 dynamic-response diagnosis, local perturbation, condition descriptor 한계 분석 스크립트
- `notebooks/main_model/`: Colab 실행용 주요 notebook
- `results/main_model/`: FLOW, CNF, MIXPRIOR/HTBAL 계열 최종 모델 비교 결과 csv/json
- `results/context_ablation/`: context ablation 및 claim 검토 결과 csv/json/png
- `reports/inverse_design_yield_benchmark_kr.md`: 최종 모델 simulator benchmark 요약
- `reports/dynamic_analysis/dynamic_response_summary.md`: dynamic-response diversity 해석 요약
- `reports/dynamic_analysis/lambda35_audit/`: lambda=35 기준 D-family/S-family 진단 그림, 표, 보조 보고서
- `reports/context_ablation/01_context_model_performance_report.md`: context ablation 결과 보고서
- `reports/claim_validation/final_claim_synthesis.md`: 초기 C1-C3 주장과 후속 분석 종합

## 프로젝트 개요

본 프로젝트는 FMO-like Hamiltonian의 inverse design 문제를 다룹니다. 목표는 에너지 전달 조건 또는 transfer behavior가 주어졌을 때, 그 조건을 만족할 가능성이 높은 Hamiltonian `H`를 생성하는 것입니다.

초기에는 조건부 normalizing flow가 `p(H | c)`를 잘 학습하면 특정 효율 조건에 맞는 Hamiltonian을 생성할 수 있을 것이라고 보았습니다. 그러나 실험을 진행하면서 단순 likelihood나 scalar condition만으로는 충분하지 않다는 점이 확인되었습니다. 같은 high-transfer 범위 안에서도 Hamiltonian들은 서로 다른 trajectory response를 보였고, 정적인 Hamiltonian feature나 몇 개의 scalar label만으로는 이 다양성을 안정적으로 지정하기 어려웠습니다.

따라서 최종 방향은 다음과 같이 정리됩니다.

1. 조건부 생성 모델로 Hamiltonian 분포를 학습한다.
2. 생성된 Hamiltonian을 다시 simulator로 검증한다.
3. 단순 likelihood뿐 아니라 target-match yield, support, dynamic-response diversity를 함께 평가한다.
4. scalar condition만으로 부족한 dynamic 다양성을 어떻게 보존할 수 있는지 분석한다.

최종 모델 비교에서는 FLOW 계열 모델을 주된 inverse-design yield 후보로 두고, CNF baseline 및 MIXPRIOR/HTBAL 계열 모델은 density, support, diversity trade-off를 해석하기 위한 비교 축으로 사용했습니다.

## 연구 흐름

### 1. 데이터와 기본 문제 설정

Hamiltonian `H`는 FMO-like system에서 가능한 site energy와 coupling을 나타내는 행렬입니다. 각 Hamiltonian은 simulator를 통해 효율, transfer time, site population trajectory, trap/loss behavior 등의 label을 얻습니다. 생성 모델은 이 데이터로부터 조건부 분포 `p(H | c)`를 학습합니다.

이때 중요한 점은, 하나의 condition이 하나의 Hamiltonian만을 결정하지 않는다는 것입니다. 같은 효율 또는 같은 transfer target을 만족하는 Hamiltonian이 여러 개 존재할 수 있으므로, 이 문제는 many-to-one inverse design 문제에 가깝습니다.

### 2. 초기 claim 검토

초기에는 high-efficiency Hamiltonian이 정적인 물리 feature space에서 뚜렷한 cluster를 이룰 수 있다고 보았습니다. 그러나 raw Hamiltonian, eigenvalue/IPR 기반 feature, site-fixed feature 등을 바꿔가며 clustering을 시도한 결과, high-eta subset만의 안정적인 discrete cluster 구조라고 보기에는 근거가 약했습니다. clustering score는 feature와 seed에 민감했고, random subset에서도 비슷한 구조가 나타나는 경우가 있었습니다.

따라서 최종 해석은 “정적 feature만으로 high-efficiency Hamiltonian family를 강하게 분리할 수 있다”가 아니라, “정적 feature만으로는 dynamic response 다양성을 충분히 설명하기 어렵다”에 가깝습니다.

### 3. Context ablation

조건 `c`에 어떤 정보를 넣는지가 모델 성능에 미치는 영향을 확인하기 위해 27D Hamiltonian 생성 문제는 유지하고, context feature만 바꾸어 NSF 모델을 학습했습니다.

| context | dim | best val NLL | 해석 |
| --- | ---: | ---: | --- |
| c5 | 5 | 21.96 | 원래 scalar condition에 가까운 기본 설정 |
| c12 | 12 | 15.70 | eigenvalue 등 Hamiltonian-derived 정보 추가 |
| c18 | 18 | 19.44 | dynamic summary 일부 추가, 단순한 상하관계는 아님 |
| c25 | 25 | 14.18 | 더 많은 summary를 넣었지만 항상 단조 개선되지는 않음 |
| c26 | 26 | 8.91 | population trajectory 계열 정보가 강하게 작용 |
| c33 | 33 | 6.12 | 가장 많은 context 정보 사용 |

c26, c33은 validation NLL이 크게 낮았지만, 이들은 trajectory-derived 정보를 포함하므로 원래의 5D scalar condition보다 훨씬 강한 조건을 제공한 결과로 해석해야 합니다. 즉 “모델이 모든 의미에서 더 좋아졌다”라기보다, condition에 담긴 정보량이 증가하면 Hamiltonian 분포를 더 좁게 지정할 수 있다는 결과입니다.

### 4. Dynamic-response 분석

FMO inverse design에서 중요한 것은 단순히 높은 eta를 얻는 것이 아니라, 어떤 방식으로 에너지가 이동하는지도 함께 보는 것입니다. 같은 high-transfer 범위 안에서도 trap 도달 양상, source retention, residual decay 등이 달라질 수 있습니다.

이를 위해 두 종류의 reference family를 사용했습니다.

- `D-family` (`D000`-`D012`): 각 Hamiltonian을 `lambda=35` 조건에서 simulator에 넣어 얻은 0-50 ps transfer trajectory를 기준으로 나눈 dynamic-response family입니다. 입력 feature는 `eta(t)`, `d eta/dt`, site-group population trajectory, transfer-time/residual/loss summary 등을 합친 trajectory-derived embedding입니다.
- `S-family` (`S000`-`S037`): Hamiltonian 자체에서 얻은 structural embedding을 기준으로 나눈 structural mode입니다. 이는 trajectory를 직접 사용하지 않는 구조적 reference label이며, D-family와의 관계를 보조적으로 확인하는 데 사용했습니다.

여기서 family는 독립적인 물리 mechanism을 증명하는 강한 cluster claim이 아니라, 생성 모델이 다양한 transfer behavior를 보존하는지 진단하기 위한 reference lens로 사용됩니다.
구체적인 D-family/S-family 진단 그림과 표는 `reports/dynamic_analysis/lambda35_audit/`에 함께 정리했습니다.

관련 분석에서는 다음 점을 확인했습니다.

- 정적인 Hamiltonian feature만으로는 dynamic response 차이를 안정적으로 설명하기 어렵다.
- trajectory-derived compact readout은 transfer behavior 차이를 더 직접적으로 표현한다.
- S-family와 D-family는 무작위보다 높은 관련성을 보이지만, 강한 1:1 대응은 아니므로 구조 label만으로 dynamic behavior를 지정하기는 어렵다.

### 5. 최종 모델 평가

최종 평가는 likelihood만 보지 않고, 생성된 Hamiltonian을 simulator에 다시 넣어 target 조건을 만족하는지 확인하는 방식으로 진행했습니다. 이 평가는 FLOW 계열 모델을 중심으로 보되, CNF baseline과 MIXPRIOR/HTBAL 변형을 함께 비교하여 yield, density, support, diversity 사이의 절충 관계를 확인합니다.

대표 benchmark 결과는 다음과 같습니다.

| model | mean target-match rate | valid designs per 512 | 해석 |
| --- | ---: | ---: | --- |
| CNF baseline | 0.590 | 301.9 | 기본 조건부 생성 baseline |
| CNF WMODE | 0.572 | 292.7 | mode prior 변형, 평균 yield는 개선되지 않음 |
| FLOW HTBRANCHPINNTRAJ | 0.669 | 342.4 | 평균 target-match yield가 가장 높음 |
| HTBAL CNF MIXPRIOR | 0.645 | 330.0 | late-high와 support-conservative 비교에서 강점 |

target별로 보면 FLOW HTBRANCHPINNTRAJ는 fast-high와 very-fast target에서 강했고, HTBAL CNF MIXPRIOR는 late-high target과 support-filtered 평가에서 더 안정적인 면이 있었습니다. 따라서 하나의 모델이 모든 기준에서 압도적으로 우월하다고 보기보다는, inverse-design yield와 support/density 안정성을 함께 비교하는 것이 적절합니다.

## 핵심 결론

1. Static Hamiltonian feature만으로 high-efficiency Hamiltonian의 robust discrete cluster를 주장하기는 어렵습니다.
2. 조건부 생성 모델은 likelihood fit만으로 평가하면 부족하며, simulator 기반 target-match 검증이 필요합니다.
3. context에 trajectory-derived 정보를 넣으면 분포를 더 잘 지정할 수 있지만, 이는 더 강한 target descriptor를 제공한 효과로 해석해야 합니다.
4. 최종 모델 비교에서는 FLOW HTBRANCHPINNTRAJ가 평균 target-match yield에서 가장 좋았고, HTBAL CNF MIXPRIOR는 support와 일부 target에서 더 보수적인 장점을 보였습니다.
5. 프로젝트의 최종 메시지는 “물리 메커니즘을 완전히 설명했다”가 아니라, “FMO Hamiltonian inverse design에서는 condition과 dynamic-response diversity를 함께 고려해야 하며, 생성 모델은 이를 simulator 검증으로 평가해야 한다”입니다.

## 폴더 구조

```text
20261R0136COSE362/
├─ README.md
├─ requirements.txt
├─ src/
├─ scripts/
├─ notebooks/
├─ reports/
├─ results/
└─ docs/
```

## 폴더 설명

### `src/`

재사용 가능한 Python 모듈입니다.

- `src/fmo_hamiltonian/`: Hamiltonian sampling, simulator wrapper, constants, trajectory feature utilities
- `src/fmo_context_ablation/`: context feature 구성, H27 utilities, NSF model, data loading utilities

### `scripts/`

학습, 평가, 진단 실험을 실행하는 스크립트입니다.

- `scripts/main_model/`: FLOW 계열 최종 모델 학습, model-selection audit, inverse-design yield benchmark
- `scripts/context_ablation/`: context ablation 학습/평가 및 baseline 비교
- `scripts/final_report_experiments/`: 최종 발표자료 2-6쪽의 진단 실험 코드
- `scripts/diagnostics/`: modality, D-family separation, surrogate check 등 추가 진단 스크립트

### `notebooks/`

일부 모델 실행을 위한 Colab-oriented notebook입니다.

### `reports/`

최종 PPT 보고서 작성에 사용한 markdown 보고서와 주요 figure 자료입니다.

- `reports/context_ablation/`: context ablation, nearest-neighbor baseline, original claim validation 보고서
- `reports/context_claims_summary/`: claim 요약 그림과 markdown
- `reports/claim_validation/`: 초기 C1-C3 주장 검토와 후속 분석 정리
- `reports/dynamic_analysis/`: dynamic-response diversity 관련 제출용 요약

### `results/`

csv, json, png 형식의 compact 결과 파일입니다.

- `results/main_model/`: model-selection audit 및 inverse-design benchmark evidence
- `results/context_ablation/`: context ablation 및 baseline comparison output

### `docs/`

방법론 설명과 재현을 돕는 문서입니다.

먼저 보면 좋은 파일은 다음과 같습니다.

- `docs/main_model/MODEL_STRUCTURE_AND_RESULT_LOGIC_KR.md`
- `docs/method_notes/dynamic_clustering_method_summary.md`
- `docs/method_notes/sample_H_geom_sampling_method.md`
- `docs/method_notes/q_range_distribution_validation.md`

## 실행 환경

공통 의존성은 다음 명령으로 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

GPU 환경에서 실행할 경우, 사용하는 런타임에 맞는 PyTorch build를 먼저 설치한 뒤 학습 스크립트를 실행하면 됩니다. 이 저장소는 특정 로컬 GPU backend를 전제로 하지 않습니다.

## 대용량 데이터 파일

GitHub 저장소에는 용량 문제로 raw dataset을 포함하지 않았습니다. 대용량 데이터 묶음은 repository 밖의 `dataset/` 폴더로 따로 정리했습니다. 

- Google Drive dataset bundle: https://drive.google.com/drive/folders/15odknCEOtuv5BGfLg7SBWOlc4Y3qlmVy?usp=drive_link

| 용도 | 필요한 파일 | 권장 배치 경로 | 비고 |
| --- | --- | --- | --- |
| 140k Hamiltonian/context dataset | `merged_h27_140k.npz` | `data/merged_h27_140k.npz` | context ablation, original claim validation, NSF baseline 학습에 사용 |
| 62k trajectory dataset | `pilot_raw_62k_lambda35.npz` | `outputs/pilot_sampling/pilot62000_t50_schema_v2_20260603_merged/pilot_raw.npz` | lambda=35에서 생성한 full trajectory 기반 데이터. D-family/dynamic analysis의 원본 |

현재 repository에 포함된 `reports/`, `results/`의 csv/json/png 파일은 위 데이터셋에서 계산된 compact evidence입니다. 따라서 보고서의 핵심 수치와 그림은 repository 안에서 확인할 수 있고, raw data 수준에서 다시 분석하려면 위 두 데이터 파일을 추가로 받아 배치하면 됩니다.

## 확인 및 부분 재현 순서

전체 raw dataset은 용량 문제로 포함하지 않았기 때문에, 이 저장소는 다음 두 방식으로 확인하는 것을 권장합니다.

1. 결과 검토: `reports/`, `results/`, `docs/`의 compact evidence를 읽어 최종 결론과 수치를 확인합니다.
2. 코드 검토: `src/`와 `scripts/`에서 학습, 평가, 진단 실험이 어떤 방식으로 구성되었는지 확인합니다.

권장 읽기 순서는 다음과 같습니다.

1. `docs/main_model/MODEL_STRUCTURE_AND_RESULT_LOGIC_KR.md`
2. `reports/inverse_design_yield_benchmark_kr.md`
3. `reports/dynamic_analysis/dynamic_response_summary.md`
4. `docs/method_notes/dynamic_clustering_method_summary.md`
5. `reports/context_ablation/01_context_model_performance_report.md`
6. `reports/context_ablation/03_nearest_neighbor_baseline_report.md`
7. `reports/context_ablation/04_original_claim_validation_report.md`
8. `reports/claim_validation/final_claim_synthesis.md`

## 제외한 파일

GitHub 저장소가 과도하게 커지는 것을 막기 위해 다음 파일은 제외했습니다.

- raw `.npz` dataset
- virtual environment 및 cache folder
- 중복 archive 파일
- 임시 local output

대신 결과 확인과 보고서 작성 근거 검토에 필요한 작은 csv/json/png/md 파일은 포함했습니다.
