# Original C1-C3 Claim Validation Audit

## 한 줄 결론

초기 proposal의 C1-C3를 그대로 강하게 주장하기는 어렵다. C1과 C3의 강한 형태는 현재 결과로 지지되지 않았고, C2는 source-sink delocalization 신호는 보이지만 bath-resonance 조건까지 동시에 대부분 만족한다는 강한 형태는 지지되지 않았다. 따라서 최종 보고서에서는 "초기 claim을 모두 입증했다"가 아니라, "초기 claim을 검증했고, 그 결과 condition 표현과 baseline 선택의 중요성으로 연구 질문을 정제했다"는 흐름이 가장 안전하다.

| claim | 원래 주장 | 현재 판정 | 핵심 근거 |
|---|---|---|---|
| C1 | high-efficiency Hamiltonian은 여러 cluster/family로 나뉜다 | 강한 형태 기각 | physical feature clustering에서 silhouette delta가 거의 0이고 bootstrap CI가 겹침 |
| C2 | high-efficiency Hamiltonian은 bath resonance와 input-sink delocalized eigenstate를 동시에 만족한다 | 부분 지지 | delocalization은 high-eta에서 강하지만 bath resonance는 enrichment가 약함 |
| C3 | 실제 FMO Hamiltonian은 학습된 `p(H \| c_FMO)`에서 top-likelihood sample이다 | 기각 | generated baseline 기준 FMO percentile이 최고 10.8%로 top 5% 기준에 크게 못 미침 |

---

## 1. C1: high-eta Hamiltonian은 여러 cluster/family로 나뉘는가?

### 원래 주장

C1의 핵심은 다음 질문이다.

> 높은 transfer efficiency를 만드는 Hamiltonian들이 하나의 연속적인 영역이 아니라, 서로 구분 가능한 여러 family 또는 cluster를 이루는가?

이 주장은 단순히 전체 Hamiltonian 분포가 어딘가에서 나뉜다는 뜻이 아니다. C1이 성립하려면 high-eta subset에서 관찰되는 cluster structure가 random subset이나 전체 데이터셋의 일반적인 구조보다 더 뚜렷해야 한다.

### 검증 방식

raw 27D Hamiltonian을 그대로 clustering하면 site permutation과 source/sink 역할 문제가 섞인다. 그래서 C1 검증에서는 raw H 대신 H에서 계산되는 physical feature를 사용했다.

대표 실험은 다음 feature를 사용했다.

| feature block | dim | 의미 |
|---|---:|---|
| sorted eigenvalues | 7 | Hamiltonian의 energy spectrum |
| sorted eigenstate IPR | 7 | eigenstate localization pattern |
| total | 14 | label-derived `c_l1`은 제외 |

판정 기준은 발표자료의 강한 기준에 맞췄다.

```text
delta = silhouette(high-eta or model) - silhouette(random)
accept if delta > 0.1 and bootstrap CI does not overlap
```

### 결과

`experiments/claim_validation/exp02a_c1_physical_cluster.md`에 정리된 대표 결과는 다음과 같다.

| k | model silhouette | random silhouette | delta |
|---:|---:|---:|---:|
| 2 | 0.199 | 0.198 | +0.0005 |
| 3 | 0.143 | 0.161 | -0.018 |
| 4 | 0.141 | 0.149 | -0.008 |
| 5 | 0.132 | 0.151 | -0.019 |
| 6 | 0.134 | 0.131 | +0.004 |

best k인 `k=2`에서도 delta는 `+0.0005`로 사실상 0에 가깝다.

Bootstrap CI도 분리되지 않았다.

| group | k | mean | 95% CI |
|---|---:|---:|---:|
| model/high-eta side | 2 | 0.201 | [0.190, 0.215] |
| random baseline | 2 | 0.200 | [0.185, 0.216] |

추가로 source/sink를 고정하고 나머지 site를 정렬한 H-only feature에서는 일부 k에서 약한 양의 delta가 나왔지만, seed에 민감했다. 예를 들어 k=2 delta는 seed 716에서 `+0.0362`였지만 seed 717에서는 `-0.0095`로 방향이 뒤집혔다.

### 해석

C1의 강한 형태는 현재 결과로 지지되지 않는다.

다만 이것이 "어떤 구조도 없다"는 뜻은 아니다. dynamics-based clustering에서는 family-like structure가 관찰된 적이 있다. 그러나 그 결과는 C1의 강한 판정 기준인 "high-eta subset이 random baseline보다 뚜렷하게 cluster된다"를 만족한 것은 아니다. 따라서 최종 보고서에서는 C1을 다음처럼 쓰는 것이 안전하다.

> High-efficiency Hamiltonian이 discrete structural clusters를 이룬다는 강한 C1은 현재 physical-feature silhouette 기준으로 지지되지 않았다. 일부 dynamics-based family structure는 관찰되었지만, high-eta만의 robust cluster evidence로 보기에는 부족하다.

---

## 2. C2: high-eta Hamiltonian은 mechanistic signature를 갖는가?

### 원래 주장

C2는 high-eta Hamiltonian이 다음 두 조건을 동시에 만족한다는 주장이다.

1. eigenvalue gap이 bath spectrum과 잘 맞는다.
2. input site와 sink site를 동시에 포함하는 delocalized eigenstate가 존재한다.

여기서 input site는 site 1, sink site는 trap으로 빠지는 site 3이다.

### 검증 방식

C2는 기존 140k dataset H에 대해 추가 시뮬레이션 없이 계산할 수 있다. 각 Hamiltonian을 고유분해해서 eigenvalue와 eigenvector를 얻고, 다음 두 점수를 계산한다.

| 점수 | 정의 | 해석 |
|---|---|---|
| bath resonance score | 모든 eigenvalue gap 중 Drude-Lorentz bath spectrum `S(ΔE)`가 가장 큰 값 | 특정 energy gap이 bath noise와 잘 맞는지 |
| source-sink delocalization score | 각 eigenstate에서 site 1 weight와 site 3 weight의 harmonic mean 중 최댓값 | 하나의 eigenstate가 input과 sink 양쪽에 동시에 걸쳐 있는지 |

두 점수의 절대 threshold는 물리적으로 고정되어 있지 않으므로, 우선 dataset 내부 75 percentile을 기준으로 strong signature 여부를 정했다.

```text
bath_pass  = bath_score  >= dataset 75th percentile
deloc_pass = deloc_score >= dataset 75th percentile
joint_pass = bath_pass and deloc_pass
```

이 기준은 "상위 25% 수준으로 강한 bath resonance와 상위 25% 수준으로 강한 source-sink delocalization을 동시에 만족하는가"를 보는 것이다. 강한 C2가 맞다면 high-eta group에서 joint pass rate가 매우 높아야 한다.

사용 스크립트:

```powershell
python scripts/eval_c2_mechanistic_signature.py
```

### 결과

140k dataset 전체에 대해 계산한 결과는 다음과 같다.

| group | n | eta median | bath score median | deloc score median | bath pass | deloc pass | joint pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 140000 | 0.656 | 0.998 | 0.020 | 25.0% | 25.0% | 6.6% |
| high eta >= 0.95 | 19194 | 0.974 | 0.997 | 0.156 | 22.2% | 70.2% | 16.2% |
| nonhigh eta < 0.85 | 96559 | 0.461 | 0.998 | 0.010 | 25.7% | 10.7% | 3.4% |
| top 10% eta | 14000 | 0.979 | 0.997 | 0.169 | 21.8% | 73.3% | 16.6% |
| bottom 50% eta | 70000 | 0.340 | 0.998 | 0.006 | 25.9% | 6.7% | 2.3% |

### 해석

C2의 두 요소는 서로 다르게 나타났다.

첫째, source-sink delocalization은 high-eta에서 강하다. 전체 기준 deloc pass rate는 25.0%인데, high-eta에서는 70.2%로 크게 올라간다. nonhigh group에서는 10.7%에 불과하다. 즉 high-eta Hamiltonian은 input site와 sink site를 같은 eigenstate 안에서 동시에 포함하는 경향이 강하다.

둘째, bath resonance score는 high-eta에서 특별히 높아지지 않는다. bath pass rate는 전체가 25.0%, high-eta가 22.2%로 오히려 약간 낮다. bath score median도 all 0.998, high-eta 0.997로 거의 차이가 없다. 따라서 현재 정의의 bath resonance는 high-eta를 구분하는 강한 signature로 보이지 않는다.

셋째, 두 조건을 동시에 만족하는 joint pass rate는 high-eta에서 16.2%로 전체 6.6%보다 높다. 하지만 강한 C2가 요구하는 "high-eta 대부분이 두 조건을 동시에 만족한다"와는 거리가 멀다.

따라서 C2는 다음처럼 정리하는 것이 안전하다.

> High-eta Hamiltonian은 source-sink delocalized eigenstate를 갖는 경향이 강하게 나타났다. 그러나 bath resonance 조건은 high-eta에서 뚜렷하게 enrichment되지 않았고, 두 조건을 동시에 만족하는 비율도 16.2%에 그쳤다. 따라서 C2의 강한 형태는 지지되지 않지만, input-sink delocalization은 high-efficiency transfer의 유의미한 mechanistic signature 후보로 남는다.

### 주의점

이 검증에서 bath resonance score는 "Drude-Lorentz spectrum이 큰 eigenvalue gap이 있는가"로 정의했다. 그러나 실제 Redfield dynamics에서는 단순히 gap 하나가 bath spectrum peak에 가깝다는 것만으로 transfer 효율이 결정되지 않는다. eigenstate overlap, site-bath coupling operator, trap/loss rate, energy funnel 등이 함께 작용한다. 그러므로 bath condition이 약하게 나온 것은 C2의 표현이 너무 단순했기 때문일 수도 있다.

---

## 3. C3: 실제 FMO H는 학습 분포에서 top-likelihood sample인가?

### 원래 주장

C3의 강한 형태는 다음과 같다.

> 실제 FMO Hamiltonian은 `c_FMO` 조건에서 학습된 조건부 분포 `p(H | c_FMO)` 안에서 높은 likelihood를 가진다.

여기서 `c_FMO`는 실제 FMO Hamiltonian을 simulator로 돌려 만든 condition이다.

### 검증 방식

두 baseline을 구분했다.

| baseline | 비교 대상 H | 의미 |
|---|---|---|
| dataset baseline | 기존 140k dataset H를 `c_FMO` 조건에서 scoring | FMO가 전체 dataset H보다 그럴듯한가 |
| generated baseline | `c_FMO`를 넣고 모델이 새로 생성한 H를 scoring | FMO가 모델의 실제 생성분포 안에서 typical/top sample인가 |

C3의 직접 검증은 generated baseline이다. 왜냐하면 주장은 "학습된 생성분포 안에서 FMO가 높은 likelihood인가"이기 때문이다.

### 결과

generated baseline 기준 FMO percentile은 다음과 같다.

| context | generated baseline FMO percentile | 판정 |
|---|---:|---|
| c5 | 5.95% | top 5% 기준 불만족 |
| c12 | 4.12% | top 5% 기준 불만족 |
| c18 | 0.00% | top 5% 기준 불만족 |
| c25 | 0.00% | top 5% 기준 불만족 |
| c26 | 10.79% | top 5% 기준 불만족 |
| c33 | 5.52% | top 5% 기준 불만족 |

가장 높은 값은 `c26`의 10.79%다. 이는 FMO보다 log-likelihood가 낮거나 같은 generated H가 10.79%라는 뜻이다. 반대로 말하면, generated H의 약 89%가 FMO보다 더 높은 likelihood를 가진다.

### 해석

C3의 강한 형태는 지지되지 않는다.

dataset baseline에서는 `c26/c33`에서 FMO percentile이 99% 이상으로 높게 나왔다. 그러나 이 결과는 "FMO가 기존 dataset H들보다 `c_FMO` 조건에서 그럴듯하게 평가된다"는 보조 관찰이다. generated baseline에서는 FMO가 왼쪽 tail에 놓였으므로, 모델이 실제로 `c_FMO`에서 생성하는 H들과 비교했을 때 FMO가 top-likelihood sample이라고 말할 수 없다.

---

## 4. 최종 정리

초기 C1-C3 claim을 그대로 성공 claim으로 쓰기는 어렵다.

| claim | 강한 형태 | 현재 결론 | 최종보고서에서의 안전한 표현 |
|---|---|---|---|
| C1 | high-eta H는 여러 discrete cluster로 나뉜다 | 기각/약함 | high-eta cluster 구조는 silhouette와 baseline 기준에서 robust하지 않았다 |
| C2 | high-eta H는 bath resonance와 delocalized eigenstate를 동시에 만족한다 | 부분 지지 | source-sink delocalization은 강한 신호지만 bath resonance는 약했다 |
| C3 | FMO H는 generated distribution의 top-likelihood sample이다 | 기각 | FMO는 dataset baseline에서는 높지만 generated baseline에서는 top-likelihood가 아니다 |

이 결과를 최종보고서에서 부정적으로만 쓸 필요는 없다. 오히려 연구 질문이 다음처럼 정제되었다고 쓰는 편이 좋다.

> 초기에는 high-efficiency Hamiltonian의 discrete family, mechanistic signature, biological FMO likelihood를 강한 claim으로 설정했다. 그러나 실험 결과 C1/C3의 강한 형태는 지지되지 않았고, C2도 source-sink delocalization만 부분적으로 지지되었다. 이 negative result를 바탕으로 우리는 조건부 생성 문제에서 condition representation과 baseline choice가 결론을 크게 바꾼다는 더 방어 가능한 결론으로 연구 방향을 정제했다.

