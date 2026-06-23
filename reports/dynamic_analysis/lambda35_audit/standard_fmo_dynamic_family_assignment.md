# 표준 FMO의 D-family 사후 배정

이 문서는 표준 FMO Hamiltonian이 trajectory 기반 `D000`-`D012` family 중 어디에 가까운지 확인한 별도 검증 노트다. 목적은 dynamic diversity가 왜 중요한지를 직접 설명하는 것이 아니라, D-family 분석이 단순히 데이터 내부의 임의적 분류인지, 아니면 실제 생물학적 reference Hamiltonian과도 어느 정도 연결되는지 확인하는 것이다.

따라서 이 문서는 D-family 분석의 신빙성을 보강하는 별도 appendix 또는 supporting analysis로 사용하는 것이 자연스럽다.

## 1. 질문

핵심 질문은 다음과 같다.

> 표준 FMO Hamiltonian을 D-family를 정의한 것과 동일한 dynamic feature space에 넣으면, 어떤 D family에 가장 가까운가?

여기서 주의할 점은 "표준 FMO는 eta가 높고 tau가 빠르니까 특정 D family에 속한다"가 아니라는 것이다. eta나 tau 같은 scalar summary만으로 family를 배정하면, 실제 시간 변화 패턴이 다른 Hamiltonian도 비슷하게 보일 수 있다.

따라서 표준 FMO는 다음 기준으로 비교했다.

1. 표준 FMO Hamiltonian을 같은 simulator와 같은 `lambda=35` 조건에서 시뮬레이션한다.
2. D-family clustering에 사용된 것과 동일한 dynamic feature를 추출한다.
3. 이 feature vector를 D-family centroid 및 nearest member와 비교한다.

## 2. 비교에 사용한 dynamic feature

표준 FMO를 비교할 때 사용한 feature는 단순 scalar label이 아니다. D-family를 정의할 때 사용한 dynamic embedding과 같은 구조를 사용했다.

포함된 정보는 다음과 같다.

- eta(t) trajectory
- d eta/dt trajectory
- path-group trajectory
- trap, loss, residual 관련 시간 정보
- sink34, detour567 residence 관련 요약값
- eta10, eta20, eta50, tau_transfer_est 같은 scalar summary

즉 이 비교는 "최종 효율이 비슷한가"가 아니라 "시간에 따른 transfer pattern이 어떤 D family와 가장 비슷한가"를 보는 것이다.

## 3. D-family별 표준 FMO와의 거리

표준 FMO를 전체 dynamic feature space에 사후 투영한 결과, centroid 기준과 nearest member 기준 모두에서 `D003`이 가장 가까웠다.

| D family | centroid distance | nearest member distance | 해석 |
|---|---:|---:|---|
| `D003` | 8.40 | 2.61 | 표준 FMO와 가장 가까운 dynamic family. |
| `D010` | 10.41 | 5.24 | 두 번째로 가깝지만 D003보다 전체 trajectory 거리가 크다. |
| `D002` | 15.94 | 6.22 | delayed-high transfer 쪽으로, FMO와는 더 멀다. |
| `D011` | 19.54 | 14.79 | 최종 효율은 높을 수 있지만 trajectory pattern은 FMO와 다르다. |
| `D004` | 21.86 | 14.24 | intermediate family로, FMO와는 거리가 크다. |
| `D009` | 22.75 | 10.46 | scalar summary는 FMO와 비슷해 보일 수 있지만 전체 trajectory 기준으로는 멀다. |

이 결과에서 가장 중요한 비교는 `D003`과 `D009`다. `D009`는 eta, tau, loss 같은 scalar summary만 보면 표준 FMO와 비슷한 fast high-efficiency family처럼 보인다. 하지만 전체 trajectory feature를 사용하면 D003보다 훨씬 멀다.

따라서 표준 FMO가 D003에 가깝다는 결론은 단순히 eta가 높거나 tau가 짧아서 나온 것이 아니다. 전체 시간 변화 패턴을 기준으로 비교했을 때 D003이 가장 가까웠다는 뜻이다.

## 4. D003 family는 어떤 특성을 가지는가

표준 FMO가 가장 가까웠던 `D003`는 전체 데이터의 약 9.6%를 차지하는 high-efficiency dynamic family다. D003의 대표적인 median 지표와 표준 FMO의 값을 나란히 놓으면 다음과 같다.

| metric | D003 median | standard FMO | 해석 |
|---|---:|---:|---|
| eta20 | 0.823 | 0.970 | 둘 다 20 ps 시점부터 높은 transfer를 보이지만, 표준 FMO가 더 빠른 쪽에 있다. |
| eta50 | 0.945 | 0.994 | 둘 다 최종적으로 high-efficiency 영역에 남는다. |
| tau_transfer_est | 8.935 | 6.450 | 표준 FMO가 D003 median보다 더 빠르게 trap에 도달한다. |
| residence_sink34_0_10ps | 0.113 | 0.141 | 표준 FMO가 초반 sink-side population을 약간 더 크게 형성한다. |
| residence_detour567_0_10ps | 0.187 | 0.091 | 표준 FMO는 D003 median보다 detour 영역 residence가 작다. |
| loss_50ps | 0.011 | 0.006 | 둘 다 loss가 낮고, 표준 FMO가 더 낮다. |

따라서 D003는 "가장 빠른 family"라기보다는, 초반부터 비교적 높은 transfer를 보이고 최종 효율도 높은 high-efficiency family로 해석하는 편이 적절하다. `D009`와 `D012`는 eta20이 더 높고 tau가 더 짧은 fast high-efficiency family에 가깝다. 반면 D003는 그보다 약간 완만하지만, loss가 낮고 최종 eta가 높은 안정적인 high-efficiency family다.

표준 FMO의 scalar summary는 D003 median보다 더 빠르고 더 높은 편이다. 즉 표준 FMO가 D003의 평균적인 샘플과 완전히 같다는 뜻은 아니다. 오히려 표준 FMO는 D003 family에 가장 가까운 trajectory pattern을 가지면서, scalar 성능 면에서는 D003 내부의 빠른 쪽 또는 D009/D012 쪽 특성도 일부 갖는 reference로 보는 것이 안전하다.

또 하나 주의할 점은 D003가 특정 structural mode 하나로 강하게 설명되는 family는 아니라는 점이다. D003의 가장 큰 structural mode는 `S001`이지만, 그 비율은 약 5.7%에 불과하다. 따라서 D003를 "특정 구조 하나의 family"라고 해석하기보다는, 여러 구조적 배경 위에서 비슷한 dynamic response가 나타난 family로 보는 편이 더 맞다. 이 점도 D-family가 Hamiltonian의 정적 구조가 아니라 trajectory-level behavior를 요약한다는 해석과 잘 맞는다.

요약하면 D003의 특성은 다음과 같다.

- high-efficiency family다.
- loss가 낮고 최종 trap 축적이 높다.
- `D009`, `D012`만큼 즉각적인 fast-transfer family는 아니지만, 20 ps부터 이미 높은 transfer를 보인다.
- detour residence가 완전히 작지는 않기 때문에, 단순 direct-transfer family로만 해석하기는 어렵다.
- 특정 structural mode 하나로 환원되기보다는, trajectory pattern 중심으로 정의된 family에 가깝다.

이 해석을 바탕으로 하면 표준 FMO 결과는 다음처럼 말할 수 있다.

> 표준 FMO는 D003의 평균적인 scalar 지표와 완전히 같은 샘플은 아니지만, 전체 trajectory pattern 기준으로 D003에 가장 가깝다. 따라서 D003는 표준 FMO-like high-efficiency trajectory를 포함하는 family로 해석할 수 있으며, 이 점은 D-family 분석이 단순한 수치 label이 아니라 실제 reference Hamiltonian과 연결될 수 있음을 보여준다.

## 5. Block-wise distance 해석

전체 거리를 feature block별로 나누면 왜 D003이 선택되는지 더 분명해진다.

| D family | total distance | eta trajectory | d eta/dt trajectory | path trajectory | summary metrics |
|---|---:|---:|---:|---:|---:|
| `D003` | 8.40 | 2.29 | 4.55 | 6.56 | 1.24 |
| `D010` | 10.41 | 3.19 | 5.10 | 8.35 | 1.55 |
| `D002` | 15.94 | 5.13 | 8.20 | 12.45 | 2.38 |
| `D011` | 19.54 | 6.39 | 10.29 | 15.10 | 2.70 |
| `D004` | 21.86 | 7.39 | 10.11 | 17.64 | 3.14 |
| `D009` | 22.75 | 7.10 | 15.33 | 15.21 | 0.53 |

`D009`의 summary metrics 거리는 0.53으로 매우 작다. 즉 요약 성능값만 보면 D009가 표준 FMO와 가까워 보일 수 있다. 하지만 d eta/dt trajectory와 path trajectory 거리가 크기 때문에 전체 dynamic pattern 기준에서는 D003보다 멀어진다.

이 점은 해석상 중요하다. 표준 FMO의 D-family 배정은 단순 성능값 비교가 아니라, trajectory shape까지 포함한 결과다.

## 6. 해석

가장 안전한 해석은 다음과 같다.

> 표준 FMO Hamiltonian은 D-family를 정의한 것과 동일한 dynamic feature space에 사후 투영했을 때 `D003`에 가장 가까웠다. 따라서 D-family는 단순히 데이터 내부에서 임의로 붙인 label만은 아니며, 실제 생물학적 reference Hamiltonian의 dynamic behavior를 어느 정도 회수하는 분석 단위일 수 있다.

다만 이 결과를 과하게 해석하면 안 된다.

- 표준 FMO가 D003의 평균적 샘플과 완전히 동일하다는 뜻은 아니다.
- D003가 표준 FMO의 물리 메커니즘을 설명한다는 뜻도 아니다.
- D-family가 엄밀한 생물학적 pathway를 발견했다는 뜻도 아니다.
- summary metric만 보면 `D009`, `D012` 같은 fast high-efficiency family와도 일부 유사성이 있다.

정확한 결론은 다음에 가깝다.

> 표준 FMO는 전체 trajectory pattern 기준으로 D003에 가장 가까운 high-efficiency dynamic reference이며, 일부 scalar summary에서는 더 빠른 high-efficiency family와도 유사성을 보인다.

## 7. 보고서에서의 사용 방식

이 결과는 dynamic diversity 자체의 필요성을 설명하는 본론보다는, D-family 분석의 신빙성을 보강하는 별도 근거로 사용하는 것이 좋다.

보고서에 넣는다면 다음 흐름이 적절하다.

1. 먼저 D-family가 high-efficiency 내부의 서로 다른 trajectory behavior를 요약한다는 점을 설명한다.
2. 그 다음 표준 FMO를 외부 reference로 사용해 D-family 공간에 사후 투영한다.
3. 표준 FMO가 전체 trajectory 기준으로 D003에 가장 가깝다는 결과를 제시한다.
4. 이를 통해 D-family가 임의적 분류만은 아니며, 실제 FMO-like dynamic behavior를 어느 정도 회수한다는 보조 근거로 사용한다.

보고서용 문장으로는 다음 정도가 적절하다.

> As an external reference check, we projected the standard FMO Hamiltonian into the same dynamic feature space used to define the D families. The standard FMO trajectory was closest to `D003` by both centroid and nearest-member distance. Importantly, this assignment was not driven only by scalar performance metrics: although `D009` had very similar summary metrics, it was much farther in eta-derivative and path-trajectory blocks. This supports the interpretation that the D-family taxonomy captures trajectory-level behavior rather than only final transfer efficiency.

한국어 보고서에서는 다음처럼 쓸 수 있다.

> 표준 FMO Hamiltonian을 D-family를 정의한 것과 동일한 dynamic feature space에 사후 투영한 결과, centroid 기준과 nearest-member 기준 모두에서 `D003`이 가장 가까웠다. 특히 `D009`는 scalar summary만 보면 FMO와 유사하지만, eta 변화율과 path trajectory block에서는 훨씬 멀었다. 따라서 이 배정은 단순히 최종 효율이 높기 때문이 아니라, 전체 trajectory pattern 기준의 결과로 해석해야 한다.

## 8. 재현 파일

이 분석에 사용한 결과 파일은 다음과 같다.

- `dynamic_diversity_audit_lambda35/tables/standard_fmo_dynamic_family_assignment.csv`
- `dynamic_diversity_audit_lambda35/tables/standard_fmo_dynamic_metrics.csv`

분석 스크립트는 다음 파일에 있다.

- `dynamic_diversity_audit_lambda35/scripts/assign_standard_fmo_to_dynamic_family.py`
