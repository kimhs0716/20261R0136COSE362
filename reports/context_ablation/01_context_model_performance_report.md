# Context Model Performance Comparison

## 한 줄 결론

`c26`이 simulator 재평가 기준에서 가장 좋은 context였다. `c33`은 validation NLL은 가장 낮았지만, 공통 5개 label의 conditional MAE 기준에서는 `c26`보다 약간 낮았다. 따라서 이번 실험의 결론은 "context 정보를 늘리면 조건부 생성은 확실히 좋아지지만, NLL 최저 모델이 항상 downstream label 재현도 최선은 아니다"에 가깝다.

## 비교 기준

모든 run은 생성 target을 27D Hamiltonian으로 고정하고, context feature set만 바꾼 NSF 모델이다.
성능 비교는 context 전체 차원이 아니라 공통 label 5개 `eta`, `tau_transfer`, `ipr`, `purity`, `c_l1`에 대해 수행한다.

- `delta = model_mae - random_mae`: 음수이면 모델이 random보다 좋음
- `reduction = 1 - model_mae / random_mae`: 양수이면 random 대비 오차 감소
- `better_fraction`: target별로 모델 오차가 random 오차보다 작은 비율

여기서 random baseline은 condition을 보지 않고 데이터 생성 prior에서 Hamiltonian을 무작위로 뽑는 기준 모델이다. NSF 모델은 target condition을 입력으로 받아 `p(H | c)`에서 H를 생성하지만, random baseline은 같은 target condition을 전혀 사용하지 않는다. 따라서 모델이 random baseline보다 좋다는 것은 단순히 그럴듯한 H를 생성했다는 뜻이 아니라, 주어진 condition에 맞는 H를 생성했다는 뜻이다.

## Context 구성

| context | 구성 | 해석상 의미 |
| --- | --- | --- |
| `c5` | labels | 원래 proposal에 가장 가까운 최소 condition |
| `c12` | labels + eigenvalues | spectrum 정보를 추가한 구조 조건 |
| `c18` | labels + dynamics summary | trajectory 전체 대신 요약 동역학 정보를 추가 |
| `c25` | labels + eigenvalues + dynamics summary | spectrum과 요약 동역학을 함께 사용 |
| `c26` | labels + selected population trajectory | selected population trajectory를 직접 조건으로 사용 |
| `c33` | labels + eigenvalues + selected population trajectory | reconstructed full-context 설정 |

## Condition 정보량과 label leakage 위험 구분

이번 ablation은 context를 일부러 여러 단계로 늘려 보았기 때문에, 성능이 높은 context가 항상 더 공정한 context라는 뜻은 아니다. 특히 simulator에서 나온 trajectory 정보를 condition에 넣으면 target label을 맞추는 데 유리한 정보를 추가로 제공할 수 있다.

| 구분 | 해당 context | 해석 |
| --- | --- | --- |
| 기본 label 조건 | `c5` | 원래 5D proposal에 가장 가까운 설정 |
| H-only 구조 정보 추가 | `c12` | label에 eigenvalue만 추가하므로 trajectory-derived 정보는 쓰지 않음 |
| 요약 dynamics 추가 | `c18`, `c25` | 직접 population trajectory는 아니지만 `c_l1_t`, `purity_t`, `ipr_t` 요약을 포함하므로 circularity 가능성을 별도로 봐야 함 |
| 직접 trajectory 추가 | `c26`, `c33` | selected population trajectory를 condition에 포함하므로 가장 강한 조건 |

따라서 해석은 두 층으로 나누는 것이 안전하다. 엄격하게 trajectory-derived 정보를 제외하면 `c5`와 `c12`만 비교하는 것이고, 이 경우 simulator 재평가 MAE 기준에서는 `c5`가 더 좋았다. 직접 population trajectory만 제외하고 dynamics summary까지 허용하면 `c18`이 가장 좋았다. 반면 전체 후보를 모두 포함하면 `c26`이 가장 좋다.

## 학습 조건

이번 비교는 모델 구조와 target은 고정하고 context만 바꾸는 ablation이다. 따라서 각 run의 차이는 주로 context feature set 차이로 해석한다.

| 항목 | 설정 | 이유 |
| --- | --- | --- |
| target | 27D Hamiltonian `H` | 원래 문제인 `p(H \| c)`를 유지하기 위해 생성 대상은 바꾸지 않음 |
| dataset | merged 140k samples | context별 비교에서 데이터 부족 효과를 줄이기 위해 기존 사용 가능 데이터를 합침 |
| model | NSF normalizing flow | 같은 condition에서 여러 가능한 `H`가 나올 수 있으므로 단일 회귀가 아니라 조건부 분포를 학습 |
| NSF 구조 | 8 transforms, hidden 128x2, 8 bins | 모든 context에서 같은 flow 용량을 사용해 context 효과만 비교 |
| optimizer | AdamW, lr 2e-3, weight decay 1e-4 | NSF 학습에서 안정적으로 사용한 기본 설정 |
| batch size | 2048 | 140k 데이터셋을 빠르게 학습하면서 validation 추정이 지나치게 noisy하지 않게 함 |
| gradient clipping | 1.0 | flow 학습 중 큰 gradient로 인한 불안정성을 줄임 |
| split seed | 0 | 모든 context가 동일한 train/validation split을 쓰게 해서 비교 공정성 유지 |
| max epochs | 1000 | 실제로 1000 epoch를 모두 돌리려는 설정이 아니라, 수렴 시점이 context마다 다를 수 있어 충분히 큰 상한을 둔 것 |
| early stopping | validation NLL 100 epoch 미개선 시 중단 | 과적합이 시작되거나 개선이 멈추면 자동으로 멈추고 best validation checkpoint를 사용 |
| LR scheduler | ReduceLROnPlateau, patience 20 | validation NLL 개선이 느려지면 learning rate를 낮춰 후반부 수렴을 안정화 |
| checkpoint | best validation NLL | 마지막 epoch가 아니라 validation NLL이 가장 낮았던 epoch의 가중치를 평가에 사용 |

긴 epoch 설정은 보수적인 상한이다. 실제 run들은 모두 early stopping으로 멈췄고, best epoch는 82-318 사이에 위치했다. 즉 비교에 사용된 모델은 마지막까지 과하게 진행한 가중치가 아니라, validation NLL 기준으로 가장 좋았던 시점의 checkpoint다.

## 시각화

아래 그림들은 `scripts/plot_model_performance.py`로 생성한다.

- `outputs/model_performance/comparison/figures/mean_reduction_by_context.png`: context별 평균 오차 감소율
- `outputs/model_performance/comparison/figures/label_reduction_by_context.png`: label별 random 대비 reduction
- `outputs/model_performance/comparison/figures/label_mae_by_context.png`: label별 MAE
- `outputs/model_performance/comparison/figures/nll_vs_mean_reduction.png`: validation NLL과 simulator MAE 성능의 관계
- `outputs/model_performance/comparison/figures/win_rate_by_context.png`: target-wise win rate

### 전체 평균 reduction

<img src="../outputs/model_performance/comparison/figures/mean_reduction_by_context.png" width="720">

높을수록 random 대비 조건부 생성 성능이 좋다. 전체 후보 중에서는 `c26`이 가장 높고, `c33`이 그 다음이다.

### Target-wise win rate

<img src="../outputs/model_performance/comparison/figures/win_rate_by_context.png" width="720">

각 target에서 모델이 random baseline보다 더 작은 오차를 낸 비율이다. 모든 context가 0.8 이상이므로 조건부 생성이 무조건부 random보다 안정적으로 낫다.

### Label별 reduction

<img src="../outputs/model_performance/comparison/figures/label_reduction_by_context.png" width="720">

`eta`, `tau_transfer`, `ipr`, `purity`에서는 `c26`이 가장 강하다. 반면 `c_l1`은 `c5`가 가장 높아, population trajectory 정보가 coherence label에는 항상 이득이 아님을 보여준다.

### Label별 MAE

<img src="../outputs/model_performance/comparison/figures/label_mae_by_context.png" width="720">

reduction만 보면 상대 개선율만 보이므로 실제 오차 크기도 함께 확인해야 한다. `tau_transfer`는 절대 오차 단위가 커서 다른 label과 스케일이 다르다.

### NLL과 downstream 성능

<img src="../outputs/model_performance/comparison/figures/nll_vs_mean_reduction.png" width="720">

`c33`은 validation NLL이 가장 낮지만 mean reduction은 `c26`보다 낮다. 따라서 density modeling 성능과 simulator 재평가 label 성능은 구분해서 해석해야 한다.

## Run 요약

| context | context_dim | best_epoch | stopped_epoch | stop_reason | best_val_nll | mean_reduction_fraction | mean_model_better_fraction | mean_delta_model_minus_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c5 | 5 | 82 | 182 | early_stopping | 21.9633 | 0.7173 | 0.8118 | -1.3133 |
| c12 | 12 | 203 | 303 | early_stopping | 15.7026 | 0.7021 | 0.8062 | -1.3057 |
| c18 | 18 | 123 | 223 | early_stopping | 19.4423 | 0.7254 | 0.8130 | -1.3373 |
| c25 | 25 | 227 | 327 | early_stopping | 14.1792 | 0.7048 | 0.8120 | -1.2991 |
| c26 | 26 | 291 | 391 | early_stopping | 8.9094 | 0.7742 | 0.8460 | -1.4384 |
| c33 | 33 | 318 | 418 | early_stopping | 6.1163 | 0.7645 | 0.8398 | -1.4164 |

## Label별 MAE

| context | eta_mae | tau_transfer_mae | ipr_mae | purity_mae | c_l1_mae |
| --- | --- | --- | --- | --- | --- |
| c5 | 0.0926 | 2.3046 | 0.0443 | 0.0461 | 0.0362 |
| c12 | 0.0931 | 2.3318 | 0.0488 | 0.0507 | 0.0373 |
| c18 | 0.0874 | 2.1907 | 0.0417 | 0.0436 | 0.0403 |
| c25 | 0.0931 | 2.3668 | 0.0463 | 0.0480 | 0.0404 |
| c26 | 0.0700 | 1.7206 | 0.0286 | 0.0305 | 0.0483 |
| c33 | 0.0766 | 1.8217 | 0.0304 | 0.0325 | 0.0466 |

## Label별 random 대비 reduction

| context | eta_reduction | tau_transfer_reduction | ipr_reduction | purity_reduction | c_l1_reduction |
| --- | --- | --- | --- | --- | --- |
| c5 | 0.7306 | 0.7224 | 0.6889 | 0.6681 | 0.7767 |
| c12 | 0.7292 | 0.7192 | 0.6576 | 0.6344 | 0.7700 |
| c18 | 0.7458 | 0.7362 | 0.7076 | 0.6857 | 0.7517 |
| c25 | 0.7292 | 0.7149 | 0.6750 | 0.6541 | 0.7509 |
| c26 | 0.7964 | 0.7928 | 0.7996 | 0.7801 | 0.7019 |
| c33 | 0.7771 | 0.7806 | 0.7863 | 0.7659 | 0.7124 |

## 해석

1. 모든 context가 random baseline보다 명확히 좋다. 모든 run의 평균 reduction은 약 0.70 이상이고, target별 win rate도 약 0.80 이상이다. 이는 모델이 단순히 데이터 prior에서 아무 `H`나 뽑는 수준을 넘어서, 주어진 condition에 맞는 Hamiltonian 분포를 학습했다는 1차 근거다. 
<br><br>
2. `c5`만 써도 평균 MAE가 random 대비 약 71.7% 줄었다. 즉 원래 proposal에 가까운 5D label 조건만으로도 `H -> label` 역방향 mapping에 유효한 정보가 있다. 다만 `tau_transfer`, `ipr`, `purity`처럼 dynamics에 민감한 값은 더 풍부한 context에서 개선 여지가 남아 있었다. 
<br><br>
3. trajectory-derived 정보를 엄격히 제외한 후보 중에서는 `c5`가 가장 좋았다. `c12`는 eigenvalue를 추가해 NLL은 낮췄지만, 공통 label MAE에서는 `c5`보다 낮았다. 즉 H-only spectrum 정보가 항상 target label 재현도를 높이지는 않았다. 
<br><br>
4. 직접 population trajectory를 제외하되 dynamics summary까지 허용하면 `c18`이 가장 좋았다. `c18`은 `c5`보다 평균 reduction이 높고, `eta`, `tau_transfer`, `ipr`, `purity`에서 개선을 보였다. 다만 `c18`은 simulator-derived summary를 포함하므로 원래 5D 조건보다 강한 조건이라는 점을 같이 밝혀야 한다. 
<br><br>
5. 전체 후보 중에서는 `c26`이 평균 reduction 0.7742, win rate 0.8460으로 가장 좋다. 특히 `eta`, `tau_transfer`, `ipr`, `purity`에서 가장 낮은 MAE를 보인다. selected population trajectory가 들어가면 단순 최종 label보다 시간에 따른 이동 경로 정보가 추가되므로, 같은 최종 효율을 만들 수 있는 여러 `H` 후보 중에서 더 좁은 영역을 고르게 되는 것으로 해석할 수 있다. 
<br><br>
6. `c33`은 validation NLL이 가장 낮다. 이는 `c33` context가 학습 데이터의 `p(H | c)` 밀도를 가장 잘 설명했다는 뜻에 가깝다. 그러나 simulator로 재평가한 공통 label MAE에서는 `c26`보다 약간 낮다. 따라서 NLL은 density modeling 품질 지표이고, 최종 물리 label 재현도는 별도의 downstream 지표로 봐야 한다. 
<br><br>
7. eigenvalue 추가는 NLL을 낮추는 데는 도움이 되지만, downstream label 재현 성능을 항상 높이지는 않았다. `c12`가 `c5`보다, `c25`가 `c18`보다 낮은 평균 reduction을 보인 점이 그 근거다. spectrum 정보는 Hamiltonian 구조를 설명하는 데 유용하지만, transfer label을 직접 좁히는 정보로는 population trajectory보다 약했다. 
<br><br>
8. `c_l1`은 예외적으로 `c5`에서 가장 잘 맞았다. `c_l1`은 coherence 관련 label이라 population trajectory와 같은 population 중심 정보가 항상 직접적인 이득으로 이어지지 않을 수 있다. 이 결과는 context를 많이 넣는 것이 모든 label에 무조건 좋은 것은 아니라는 점을 보여준다. 
<br><br>
9. 결론적으로 이 실험은 "NSF가 조건을 무시하지 않고 `p(H | c)`를 학습한다"는 모델 성능 검증에는 긍정적이다. 다만 어떤 context가 과학적으로 가장 정당한지는 별도 문제다. `c26`, `c33`은 성능은 좋지만 simulator-derived 정보를 많이 포함하므로, 원래 5D 문제보다 더 강한 조건을 준 실험으로 해석해야 한다.

## 한계

- 평가는 동일한 데이터 분포에서 뽑은 target condition에 대한 in-distribution 검증이다. 분포 밖 condition에 대한 일반화는 이번 평가 범위에 포함하지 않았다.
- `c26`, `c33`은 simulator-derived population trajectory를 condition에 포함하므로, 원래 5D proposal보다 더 강한 조건을 제공한다. 따라서 성능 향상은 condition 정보량 증가의 효과로 해석해야 한다. 이 점을 고려하면 엄격한 non-trajectory 조건에서는 `c5`, 직접 trajectory를 제외한 조건에서는 `c18`, 전체 조건에서는 `c26`이 각각 가장 좋은 결과로 정리된다.
- random baseline은 데이터 생성 prior에서 뽑은 무조건부 Hamiltonian이다. 이 baseline은 condition 정보를 사용하지 않기 때문에 최소 기준에 해당한다. 더 강한 baseline, 예를 들어 nearest-neighbor retrieval baseline과 비교하면 평가가 더 엄격해질 수 있다.
