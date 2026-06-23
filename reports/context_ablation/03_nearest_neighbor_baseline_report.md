# Nearest-Neighbor Retrieval Baseline 결과 보고서

## 1. 목적

1번 모델 성능 보고서에서는 NSF 모델을 random baseline과 비교했다. random baseline은 condition을 전혀 사용하지 않는 약한 기준이다. 따라서 random baseline 대비 성능은 “모델이 condition 정보를 활용한다”는 최소 검증으로는 적절하지만, “생성 모델이 단순 검색보다 낫다”는 근거까지 주지는 않는다.

이 보고서는 더 강한 기준인 nearest-neighbor retrieval baseline을 추가로 비교한다.

> 질문: NSF는 비슷한 condition을 가진 기존 train sample을 찾아오는 nearest-neighbor baseline보다도 좋은가?

## 2. 실험 방법

각 run에서 사용한 context와 같은 context feature를 사용해 nearest neighbor를 찾았다.

1. 기존 1번 실험에서 사용한 target set을 그대로 사용한다.
2. 각 run의 checkpoint에 저장된 `y_mu`, `y_sd`로 context를 normalize한다.
3. 검색 후보는 checkpoint의 train split으로 제한한다.
4. target sample이 train split 안에 있으면 자기 자신은 제외한다.
5. 가장 가까운 train sample의 저장된 label을 nearest-neighbor baseline prediction으로 사용한다.

비교 대상은 다음 세 가지다.

| 방법 | condition 사용 | 의미 |
| --- | --- | --- |
| random baseline | 아니오 | 무조건부 Hamiltonian 샘플링 |
| nearest-neighbor baseline | 예 | 가장 비슷한 train condition의 기존 sample 검색 |
| NSF model | 예 | target condition에서 새로운 H 생성 |

평가는 기존 1번 실험과 같은 5개 label에 대해 수행했다.

```text
eta, tau_transfer, ipr, purity, c_l1
```

## 3. 전체 요약

<img src="../outputs/nearest_neighbor_baseline/comparison/figures/model_vs_nn_mean_mae.png" width="720">

**그림 해석.** 각 context에서 5개 label MAE를 평균낸 값이다. 낮을수록 좋다. 모든 context에서 nearest-neighbor baseline의 평균 MAE가 NSF보다 낮다. 특히 `c5`에서는 NN baseline이 압도적으로 강하고, `c33`에서는 두 방법의 차이가 가장 작다.

| context | model MAE 평균 | NN MAE 평균 | random MAE 평균 | model reduction vs NN | model better vs NN | NN reduction vs random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c5` | 0.5048 | 0.0582 | 1.8180 | -8.1312 | 0.0988 | 0.9687 |
| `c12` | 0.5123 | 0.3216 | 1.8180 | -0.8217 | 0.3888 | 0.8332 |
| `c18` | 0.4807 | 0.1437 | 1.8180 | -2.2450 | 0.2514 | 0.9064 |
| `c25` | 0.5189 | 0.2485 | 1.8180 | -0.9846 | 0.3842 | 0.8367 |
| `c26` | 0.3796 | 0.2482 | 1.8180 | -0.5698 | 0.4166 | 0.8528 |
| `c33` | 0.4016 | 0.3465 | 1.8180 | -0.1124 | 0.5022 | 0.7805 |

여기서 `model reduction vs NN`은 `1 - model_mae / nn_mae`다. 양수이면 모델이 NN보다 좋고, 음수이면 NN이 모델보다 좋다.

결과적으로 평균 기준에서는 모든 context에서 nearest-neighbor baseline이 NSF보다 낮은 MAE를 냈다. 즉 random baseline 대비로는 NSF가 분명히 좋지만, stronger retrieval baseline까지 이겼다고 보기는 어렵다.

다만 context별 차이는 크다. `c5`에서는 NN baseline이 압도적으로 강하고, `c33`에서는 모델과 NN의 차이가 가장 작다.

<img src="../outputs/nearest_neighbor_baseline/comparison/figures/model_reduction_vs_nn.png" width="720">

**그림 해석.** `model reduction vs NN`을 percent로 그린 것이다. 0보다 크면 NSF가 NN보다 좋고, 0보다 작으면 NN이 NSF보다 좋다. 모든 context가 0 아래에 있으므로 평균 MAE 기준에서는 NN이 더 강하다. 다만 `c33`은 -11.2%로 가장 덜 밀린다.

<img src="../outputs/nearest_neighbor_baseline/comparison/figures/model_better_fraction_vs_nn.png" width="720">

**그림 해석.** target별로 NSF가 NN보다 작은 오차를 낸 비율이다. `c5`는 9.9%에 불과하지만, `c33`은 50.2%로 거의 반반이다. 즉 `c33`에서는 평균 MAE는 NN이 낮지만, sample 단위 승률은 NSF가 완전히 밀리지는 않는다.

## 4. NN이 NSF보다 좋은 것이 이상한가

이번 결과에서 NN baseline이 NSF보다 좋은 것은 이상한 일이 아니다. 항상 당연하다고 할 수는 없지만, 이 실험 설정에서는 충분히 예상 가능한 결과다.

첫째, 평가는 in-distribution target에 대해 이루어진다. target condition은 같은 140k 데이터셋 분포에서 뽑혔고, NN baseline은 그 주변의 train sample을 직접 찾는다. 데이터셋이 충분히 조밀하면 target condition과 매우 가까운 기존 sample이 존재할 가능성이 높다.

둘째, NN baseline은 실제 train sample의 저장된 label을 그대로 사용한다. 즉 생성된 H를 다시 샘플링하면서 생기는 stochastic error가 없다. 반면 NSF는 `p(H | c)`라는 분포에서 H를 새로 샘플링하므로, 같은 condition을 넣어도 target label과 정확히 같은 H가 나오리라는 보장은 없다.

셋째, 이번 평가는 diversity나 likelihood가 아니라 label MAE만 본다. label MAE 기준에서는 “새로운 H를 생성하는 능력”보다 “비슷한 label/context를 가진 기존 sample을 정확히 찾아오는 능력”이 더 유리할 수 있다.

넷째, 일부 context는 평가 label 또는 simulator-derived 정보를 직접 포함한다. 특히 `c5`는 평가에 쓰는 5개 label 자체가 condition이고, `c26`, `c33`은 selected population trajectory까지 포함한다. 이런 경우 nearest neighbor는 단순한 약한 baseline이 아니라, label/context space에서 매우 강한 retrieval baseline이 된다.

따라서 이 결과는 NSF가 무의미하다는 뜻이 아니다. 더 정확한 해석은 다음과 같다.

| 관점 | 해석 |
| --- | --- |
| label MAE 기준 | dense dataset에서는 NN retrieval이 매우 강하다 |
| generative modeling 기준 | NSF는 기존 sample을 복사하지 않고 조건부 분포에서 새 H를 생성한다 |
| 이번 실험의 결론 | NSF는 random보다 좋지만, label MAE에서는 NN보다 우월하다고 말하기 어렵다 |

즉 NN baseline은 “생성 모델이 정말 필요한가?”를 묻는 강한 기준이다. 이번 결과는 현재 dataset과 평가 지표에서는 retrieval이 매우 경쟁력 있다는 사실을 보여준다.

## 5. 왜 `c5` NN baseline이 특히 강한가

`c5`는 다음 5개 label 자체를 condition으로 쓴다.

```text
eta, tau_transfer, ipr, purity, c_l1
```

그리고 이번 평가도 같은 5개 label의 MAE를 본다. 따라서 `c5` nearest-neighbor baseline은 사실상 “평가할 label과 가장 비슷한 train sample을 찾는” 방식이다. 이 경우 NN baseline이 매우 강해지는 것은 자연스럽다.

실제로 `c5`에서 NN baseline은 random 대비 평균 96.9%의 MAE reduction을 보였다. label별로도 NN MAE는 매우 작다.

| metric | model MAE | NN MAE | model better vs NN |
| --- | ---: | ---: | ---: |
| eta | 0.0926 | 0.0108 | 0.141 |
| tau_transfer | 2.3046 | 0.2665 | 0.085 |
| ipr | 0.0443 | 0.0041 | 0.079 |
| purity | 0.0461 | 0.0041 | 0.070 |
| c_l1 | 0.0362 | 0.0056 | 0.119 |

이 결과는 NSF가 나쁘다는 뜻이라기보다는, `c5` setting에서 label-space retrieval이 매우 강한 baseline이라는 뜻이다. 특히 140k dataset처럼 표본 수가 많으면 target label과 매우 가까운 train sample을 찾을 가능성이 높다.

<img src="../outputs/nearest_neighbor_baseline/comparison/figures/nn_reduction_vs_random.png" width="720">

**그림 해석.** NN baseline이 random baseline보다 얼마나 좋은지 보여준다. 모든 context에서 NN은 random보다 훨씬 좋고, 특히 `c5`에서는 random 대비 96.9%의 MAE reduction을 보인다. 따라서 NN은 random보다 훨씬 강한 baseline이며, 이 기준을 이기지 못했다고 해서 1번 실험의 random 대비 성능 개선이 사라지는 것은 아니다.

## 6. `c33`에서는 모델과 NN 차이가 가장 작다

`c33`은 5D label, eigenvalue, selected population trajectory를 모두 포함한 가장 강한 context다. 이 경우 NN baseline도 강하지만, 모델과의 차이는 다른 context보다 훨씬 작다.

| metric | model MAE | NN MAE | model reduction vs NN | model better vs NN |
| --- | ---: | ---: | ---: | ---: |
| eta | 0.0766 | 0.0713 | -0.0742 | 0.515 |
| tau_transfer | 1.8217 | 1.5549 | -0.1716 | 0.469 |
| ipr | 0.0304 | 0.0246 | -0.2375 | 0.488 |
| purity | 0.0325 | 0.0263 | -0.2330 | 0.469 |
| c_l1 | 0.0466 | 0.0551 | 0.1541 | 0.570 |

`c33`에서는 `c_l1`에 한해 모델이 NN보다 좋았다. `c_l1`의 model reduction vs NN은 +15.4%이고, model better fraction도 57.0%다. 반면 eta, tau, ipr, purity에서는 여전히 NN이 더 좋다.

따라서 `c33`은 “모델이 NN을 거의 따라잡는 context”라고 볼 수 있지만, 전체적으로 NN을 이겼다고 말하기는 어렵다.

<img src="../outputs/nearest_neighbor_baseline/comparison/figures/label_delta_model_minus_nn_heatmap.png" width="760">

**그림 해석.** 각 칸은 `model MAE - NN MAE`다. 빨간색과 양수는 NSF가 NN보다 나쁘다는 뜻이고, 파란색과 음수는 NSF가 NN보다 좋다는 뜻이다. 대부분의 칸이 양수지만, `c33`의 `c_l1`만 음수다. 이는 `c33`에서 coherence 관련 label은 NSF가 단순 retrieval보다 더 나은 경우가 있음을 보여준다.

## 7. context별 해석

### `c5`

NN baseline이 압도적으로 강하다. 이는 평가 label 자체를 condition으로 사용하기 때문이다. 이 결과는 `c5`에서 retrieval baseline이 매우 강하다는 사실을 보여주며, NSF가 NN보다 낫다는 주장은 할 수 없다.

### `c12`

eigenvalue를 추가했지만 NN baseline이 여전히 NSF보다 좋다. 평균 model better fraction은 38.9%다. spectrum 정보를 추가해도 retrieval이 여전히 강한 기준으로 작동한다.

### `c18`

NN baseline이 매우 강하다. 평균 NN reduction vs random은 90.6%이고, model better fraction은 25.1%다. dynamics summary가 target label과 가까운 정보를 많이 담기 때문에 검색이 잘 작동한 것으로 해석된다.

### `c25`

`c18`보다 NN과 모델 차이는 줄지만, 평균적으로 NN이 우세하다. `c_l1`에서는 모델과 NN이 거의 비슷하지만, eta/tau/ipr/purity에서는 NN이 더 좋다.

### `c26`

selected population trajectory를 포함한 context다. NN baseline은 강하지만 `c5`, `c18`만큼 압도적이지는 않다. 모델 better fraction은 평균 41.7%로 올라간다. 그래도 평균 MAE는 NN이 더 낮다.

### `c33`

전체 context 중 NSF가 NN에 가장 가까운 결과를 냈다. 평균 model better fraction은 50.2%로 거의 반반이고, `c_l1`에서는 NSF가 NN보다 좋다. 그러나 평균 MAE 기준에서는 NN이 여전히 더 낮다.

## 8. 1번 보고서 결론과의 관계

1번 보고서의 결론은 다음이었다.

> NSF는 random baseline보다 명확히 좋고, condition 정보를 활용한다.

이번 NN baseline 결과는 이 결론을 부정하지 않는다. random baseline 대비 NSF가 좋다는 사실은 여전히 유효하다.

다만 더 강한 결론은 제한된다.

> NSF가 단순 nearest-neighbor retrieval보다 우수하다고 말하기는 어렵다.

따라서 결론은 다음처럼 조정하는 것이 안전하다.

| 질문 | 답 |
| --- | --- |
| NSF가 condition 정보를 활용하는가? | 그렇다. random baseline보다 훨씬 좋다. |
| NSF가 nearest-neighbor retrieval보다 좋은가? | 평균 기준으로는 아니다. |
| 생성 모델이 완전히 불필요하다는 뜻인가? | 아니다. NN은 기존 train sample을 그대로 검색하는 방식이고, 새로운 H 분포를 smooth하게 생성하는 모델은 다른 목적을 가진다. |
| 발표/보고서에서 어떻게 써야 하나? | random baseline 결과는 condition control 근거로 쓰고, NN baseline 결과는 stronger baseline에서의 한계로 정직하게 제시한다. |

## 9. 최종 결론 문장

> Random baseline 대비 NSF는 모든 context에서 큰 폭의 MAE 감소를 보였으므로 condition-aware generation은 작동한다고 볼 수 있다. 그러나 nearest-neighbor retrieval baseline과 비교하면, 평균 MAE 기준으로는 모든 context에서 retrieval이 NSF보다 강했다. 특히 `c5`는 평가 label 자체가 condition이기 때문에 nearest-neighbor가 거의 직접적인 label-space retrieval로 작동한다. 따라서 본 실험은 NSF가 random보다 좋은 조건부 생성 모델임을 지지하지만, 현재 dataset과 평가 방식에서는 단순 retrieval baseline보다 우월하다는 강한 주장은 어렵다.

## 10. 한계

- NN baseline은 train set에 존재하는 sample을 그대로 가져오는 방식이다. 새로운 H를 생성하는 모델과 목적이 완전히 같지는 않다.
- `c5`에서는 condition과 평가 label이 동일하므로 NN baseline이 특히 유리하다.
- target set은 기존 1번 실험과 동일하지만, nearest-neighbor는 train split만 검색 후보로 사용했다. target sample이 train split 안에 있을 경우 자기 자신은 제외했다.
- 이번 비교는 label MAE 기준이다. 생성된 H의 diversity, likelihood, structural plausibility까지 평가한 것은 아니다.
