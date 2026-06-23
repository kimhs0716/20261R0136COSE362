# Dynamic Response Diversity Summary

이 문서는 FMO Hamiltonian inverse design에서 dynamic-response diversity를 왜 별도로 확인해야 하는지 정리한 제출용 요약이다.

## 배경

Hamiltonian `H`가 주어지면 simulator를 통해 시간에 따른 population, trap, loss, transfer efficiency 등을 계산할 수 있다. 같은 high-transfer 조건을 만족하는 Hamiltonian이라도 에너지가 이동하는 시간적 양상은 서로 다를 수 있다. 따라서 생성 모델을 평가할 때는 eta나 transfer time 같은 scalar metric만 보는 것보다, 생성된 Hamiltonian이 다양한 dynamic response를 재현하는지도 확인해야 한다.

## Static Feature 분석의 한계

초기 분석에서는 raw Hamiltonian 또는 eigenvalue, IPR, site-fixed coupling summary 같은 static feature를 사용해 high-efficiency Hamiltonian을 clustering했다. 그러나 이 결과는 feature 구성과 seed에 민감했고, random subset에서도 비슷한 구조가 관찰되는 경우가 있었다. 따라서 static feature만으로 robust한 high-efficiency cluster를 주장하기는 어렵다.

이 결과는 clustering 시도가 완전히 무의미하다는 뜻이 아니다. 오히려 Hamiltonian의 정적인 좌표계만으로는 transfer behavior 차이를 충분히 설명하기 어렵다는 점을 보여준다.

## Dynamic-Response Family의 역할

Dynamic-response family는 trajectory-derived feature를 기반으로 Hamiltonian들을 나눈 분석용 label이다. 여기서 family는 독립적인 물리 mechanism을 증명하는 강한 cluster claim이 아니라, 다음 질문을 확인하기 위한 진단 도구로 사용된다.

1. 같은 high-transfer 범위 안에서도 서로 다른 transfer pattern이 존재하는가?
2. scalar condition만으로 이 pattern 차이를 구분하거나 지정할 수 있는가?
3. 생성 모델이 특정 dynamic pattern으로 collapse하지 않고 여러 response 양상을 보존하는가?

따라서 D-family는 생성 모델의 coverage와 collapse를 평가하기 위한 phenotype lens에 가깝다.

## D-family와 S-family의 정의

`D-family`는 Hamiltonian matrix를 직접 clustering한 결과가 아니다. 각 Hamiltonian을 `lambda=35` simulator에 넣어 0-50 ps trajectory를 얻고, 이를 다음 feature block으로 바꾼 뒤 clustering했다.

- `eta(t)`: 0-50 ps 구간의 누적 transfer efficiency snapshot
- `d eta/dt`: transfer가 어느 시간대에 빠르게 일어나는지 나타내는 rate snapshot
- site-group population trajectory: source, site2, sink-side group, detour-like group, trap, loss, residual population의 시간별 변화
- summary metric: eta10/eta20/eta50, transfer-time proxy, early residence, final trap/loss/residual 등

이 feature들은 robust scaling 후 L2 기반 kNN graph로 변환되었고, graph community detection을 통해 `D000`-`D012` label이 붙었다. 따라서 D-family는 “비슷한 Hamiltonian 구조”가 아니라 “비슷한 simulated transfer trajectory”를 기준으로 한 reference label이다.

반면 `S-family`는 Hamiltonian 구조 embedding을 기준으로 얻은 structural mode다. 즉 trajectory를 직접 사용하지 않는 구조적 reference label이며, `S000`-`S037` 범위의 label로 정리된다. S-family는 dynamic response를 완전히 결정하는 label이 아니라, 구조적 배경과 D-family 사이의 관계를 확인하기 위한 보조 축이다. 실제 분석에서도 S-D 관계는 무작위보다 높지만 강한 1:1 대응은 아니었다.

## 최종 평가와의 연결

최종 모델 평가는 다음 세 가지를 함께 본다.

1. 생성된 Hamiltonian이 target condition을 만족하는가?
2. 생성된 Hamiltonian이 학습 데이터의 support에서 크게 벗어나지 않는가?
3. 성공 샘플이 하나의 dynamic-response family에만 몰리지 않는가?

이 기준을 사용하면 단순히 likelihood가 높은 모델보다, 실제 simulator에서 target을 만족하면서도 다양한 transfer behavior를 보존하는 모델을 더 잘 평가할 수 있다. 이 때문에 최종 모델 비교에서는 FLOW 계열 모델을 main inverse-design yield 후보로 두고, CNF baseline 및 MIXPRIOR/HTBAL 계열 모델을 density, support, diversity trade-off를 확인하는 비교 축으로 함께 사용했다.

## 요약

본 프로젝트에서 dynamic-response analysis는 물리 메커니즘을 완전히 설명하기 위한 분석이 아니라, 조건부 생성 모델이 FMO inverse design에서 필요한 transfer-behavior diversity를 보존하는지 확인하기 위한 진단 절차다. 이 관점에서 scalar condition, static Hamiltonian feature, simulator-validated dynamic response를 함께 비교했다.
