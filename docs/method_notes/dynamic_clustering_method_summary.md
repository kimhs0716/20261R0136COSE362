# Global Dynamic Family D000-D012 구성 방법

이 문서는 `D000`부터 `D012`까지의 global dynamic family가 어떤 기준으로 만들어졌는지 정리한다. 여기서 중요한 점은, 이 clustering이 Hamiltonian `H` 자체의 27차원 parameter를 직접 clustering한 것이 아니라는 점이다. 각 Hamiltonian을 simulator에 넣어 얻은 시간별 에너지 전달 trajectory를 feature로 바꾼 뒤, 그 dynamic feature space에서 비슷한 trajectory를 보이는 sample들을 묶었다.

따라서 `D000`-`D012`는 structural cluster라기보다 dynamic behavior family로 해석해야 한다. 즉 같은 `D`에 속한 Hamiltonian들은 행렬 값이 비슷하다는 뜻이 아니라, lambda=35 조건에서 forward simulation했을 때 에너지 전달 양상이 비슷하다는 뜻에 가깝다.

## 1. 왜 Hamiltonian 자체가 아니라 trajectory를 보는가

초기 분석에서는 raw Hamiltonian vector나 Hamiltonian에서 추출한 structural feature를 사용해 clustering을 시도했다. 이런 static feature들은 Hamiltonian 구조의 차이를 보여주지만, 실제 에너지 전달이 어떻게 진행되는지는 직접 담고 있지 않다. 예를 들어 두 Hamiltonian의 행렬 값이 멀리 떨어져 있어도 simulator trajectory는 비슷할 수 있고, 반대로 행렬 값이 일부 비슷해 보여도 transfer speed나 detour behavior는 다를 수 있다.

그래서 global dynamic family 분석에서는 질문을 다음처럼 바꾼다.

```text
Hamiltonian H들이 행렬 공간에서 가까운가?
```

가 아니라,

```text
Hamiltonian H들이 simulator에서 비슷한 energy-transfer trajectory를 보이는가?
```

를 본다.

이 접근은 mechanism을 직접 증명하는 것은 아니다. 다만 단순한 static clustering보다, 실제 transfer behavior를 기준으로 후보 family를 나누기 위한 방법이다.

## 2. 분석 대상과 고정 조건

global D-family는 62k pilot dataset의 real Hamiltonian sample을 대상으로 만든다. 각 Hamiltonian은 같은 simulator 설정에서 forward simulation되며, bath reorganization energy는 lambda=35로 고정한다.

사용한 dense trajectory의 시간축은 다음과 같다.

```text
time range: 0.0 ps ~ 50.0 ps
time step: 0.25 ps
number of saved time points: 201
```

다만 clustering feature를 만들 때는 모든 0.25 ps point를 그대로 쓰지 않는다. `dynamic_downsample_stride = 4`를 사용해 1 ps 간격으로 줄여서 사용한다.

```text
used time snapshots: 0, 1, 2, ..., 50 ps
number of used snapshots: 51
```

이렇게 줄이는 이유는 두 가지다.

첫째, 0.25 ps 간격의 모든 점을 그대로 쓰면 feature dimension이 커지고 가까운 시간점끼리 중복 정보가 많아진다. 둘째, 1 ps 간격만 사용해도 0-50 ps 동안의 transfer shape, fast/slow behavior, detour residence, loss/residual trend를 충분히 요약할 수 있다.

## 3. Dynamic feature의 전체 구조

각 Hamiltonian `H`는 simulator trajectory를 거쳐 하나의 dynamic feature vector로 변환된다. 이를 여기서는 `x_dyn(H)`라고 부른다.

전체 feature는 네 블록으로 구성된다.

```text
x_dyn(H)
  = efficiency trajectory block
  + efficiency rate block
  + path group trajectory block
  + summary dynamic metric block
```

차원 수는 다음과 같다.

```text
eta snapshots              51
eta rate snapshots         51
path group snapshots       51 * 7 = 357
summary dynamic metrics    13
--------------------------------
total                      472
```

즉 D-family clustering의 입력은 약 472차원의 dynamic trajectory embedding이다.

## 4. Efficiency trajectory block: eta(t)

첫 번째 블록은 `eta(t)` trajectory다. 여기서 `eta(t)`는 시간 `t`까지 trap으로 이동한 population, 즉 누적 transfer efficiency로 볼 수 있다.

0-50 ps 구간에서 1 ps 간격으로 값을 추출한다.

```text
eta(0 ps), eta(1 ps), eta(2 ps), ..., eta(50 ps)
```

이 블록의 차원은 51이다.

이 feature를 넣는 이유는 최종 효율 하나만으로는 transfer behavior를 구분하기 어렵기 때문이다. 예를 들어 두 Hamiltonian이 모두 50 ps에서 높은 eta를 보이더라도, 하나는 초반 5 ps 안에 빠르게 trap에 도달하고 다른 하나는 40 ps 이후에야 천천히 도달할 수 있다. `eta(t)` 전체 trajectory를 사용하면 이런 fast/late behavior 차이를 반영할 수 있다.

다만 이 블록이 들어간다는 것은 중요한 caveat도 만든다. D-family별 eta 차이는 완전히 독립적인 사후 검증이 아니다. 왜냐하면 eta trajectory 자체가 clustering input에 포함되어 있기 때문이다. 따라서 D-family별 eta 차이는 “clustering이 포착한 dynamic 차이의 일부”로 해석해야지, 별도의 독립 검증처럼 과장하면 안 된다.

## 5. Efficiency rate block: d eta(t) / dt

두 번째 블록은 `eta(t)`의 시간 변화율이다. 코드에서는 dense time grid에서 `np.gradient`로 `d eta / dt`를 계산한 뒤, eta trajectory와 동일하게 1 ps 간격으로 downsample한다.

```text
d eta/dt at 0 ps,
d eta/dt at 1 ps,
...
d eta/dt at 50 ps
```

이 블록의 차원도 51이다.

이 feature를 넣는 목적은 누적 eta curve의 모양을 더 민감하게 보기 위해서다. `eta(t)`는 누적값이기 때문에 시간이 지나면 plateau에 가까워질 수 있다. 반면 `d eta/dt`는 어느 시간대에 transfer가 빠르게 일어나는지, 또는 거의 멈추는지를 더 직접적으로 보여준다.

예를 들어 최종 eta가 비슷한 두 trajectory라도, 하나는 초반에 큰 rate peak를 가지고 다른 하나는 작은 rate가 길게 이어질 수 있다. 이 경우 `eta(t)`만 보면 차이가 흐려질 수 있지만, `d eta/dt`를 함께 쓰면 transfer timing 차이를 더 잘 구분할 수 있다.

## 6. Path group trajectory block

세 번째 블록은 population trajectory를 몇 개의 path group으로 묶은 값이다. 원래 simulator는 site 1-7, trap, loss의 population을 갖는다. 이를 분석용으로 다음 7개 group으로 변환한다.

```text
site1      = site 1 population
site2      = site 2 population
sink34     = site 3 + site 4 population
detour567  = site 5 + site 6 + site 7 population
trap       = trap population
loss       = recombination loss population
residual   = site 1-7에 아직 남아 있는 total population
```

각 group에 대해 0-50 ps 구간의 1 ps snapshot을 사용한다.

```text
path_group(0 ps), path_group(1 ps), ..., path_group(50 ps)
```

group이 7개이고 시간 snapshot이 51개이므로, 이 블록의 차원은 다음과 같다.

```text
51 * 7 = 357
```

이 feature를 넣는 목적은 transfer efficiency만 보는 것이 아니라, energy가 어떤 경로 후보를 거쳐 이동하는지 보기 위해서다. 예를 들어 eta가 낮은 sample이 있을 때, 그 이유가 sink 쪽으로 가지 못하고 site에 오래 남아 있기 때문인지, detour-like group에 오래 머무르기 때문인지, loss로 빠졌기 때문인지 구분할 수 있다.

여기서 `sink34`와 `detour567`은 실제 분자 구조에서 완전히 증명된 물리 경로라기보다는 본 연구의 operational grouping이다. 특히 `detour567`은 site 5, 6, 7에 머무는 population을 묶어, sink-side path와 구분되는 detour-like path group으로 정의한 것이다. 따라서 보고서에서는 “site 5/6/7 group occupancy” 또는 “detour-like group” 정도로 조심스럽게 해석하는 것이 안전하다.

## 7. Summary dynamic metric block

네 번째 블록은 긴 trajectory에서 중요한 milestone과 residence 정보를 따로 요약한 scalar feature들이다. 총 13개를 사용한다.

```text
eta10
eta20
eta50
t80
t90
tau_transfer_est
residence_sink34_0_10ps
residence_detour567_0_10ps
trap_50ps
loss_50ps
residual_50ps
sink34_50ps
detour567_50ps
```

각 항목의 의미는 다음과 같다.

`eta10`, `eta20`, `eta50`은 각각 10 ps, 20 ps, 50 ps 시점의 누적 transfer efficiency다. 최종 효율뿐 아니라 early transfer와 mid-time transfer를 함께 반영하기 위한 feature다.

`t80`, `t90`은 eta가 최종적으로 도달 가능한 수준의 80%, 90%에 도달하는 시간을 나타낸다. 이는 transfer가 빠른지 늦은지를 요약하는 timing feature다.

`tau_transfer_est`는 eta trajectory 전체를 이용해 계산한 평균적인 transfer time proxy다. 단일 시점의 eta보다 trajectory 전체의 도달 시간 구조를 더 잘 반영한다.

`residence_sink34_0_10ps`와 `residence_detour567_0_10ps`는 0-10 ps 구간에서 각각 sink34 group과 detour567 group에 population이 얼마나 머물렀는지를 나타낸다. 이 값들은 초반 transfer route를 요약하기 위해 사용한다.

`trap_50ps`, `loss_50ps`, `residual_50ps`, `sink34_50ps`, `detour567_50ps`는 50 ps 마지막 시점에서 각 group에 남아 있거나 도달한 population이다. 이들은 trajectory의 최종 상태를 요약한다.

이 summary block을 추가하는 이유는, 긴 trajectory snapshot만으로는 중요한 milestone이 feature space에서 희석될 수 있기 때문이다. 예를 들어 early transfer, final residual, early detour residence 같은 값은 해석상 중요하므로 별도 scalar로 넣어 clustering feature에 명시적으로 반영한다.

## 8. Feature normalization

위에서 만든 472차원 feature는 값의 scale이 서로 다르다. 예를 들어 eta와 path population은 대체로 0-1 범위에 있지만, `t80`, `t90`, `tau_transfer_est`는 ps 단위의 시간값이다. 이런 값을 그대로 L2 distance에 넣으면 scale이 큰 feature가 거리 계산을 지배할 수 있다.

이를 막기 위해 각 feature는 train split 기준 median과 IQR을 사용해 robust scaling한다.

```text
z_dyn(H) = (x_dyn(H) - median_train) / IQR_train
```

mean/std가 아니라 median/IQR을 쓰는 이유는 outlier에 덜 민감하게 만들기 위해서다. 특히 transfer time 계열 feature는 특정 sample에서 cap에 걸리거나 매우 큰 값을 가질 수 있으므로 robust normalization이 더 안전하다.

## 9. Distance와 graph construction

정규화된 feature vector `z_dyn(H)` 사이의 거리는 L2 distance로 본다.

```text
d(i, j) = || z_dyn(H_i) - z_dyn(H_j) ||_2
```

하지만 62k sample 전체에 대해 NxN 전체 거리 행렬을 직접 만들지는 않는다. 전체 거리 행렬은 sample 수가 커질수록 메모리와 계산량이 급격히 증가하기 때문이다.

대신 hnswlib를 사용해 approximate nearest-neighbor search를 수행하고, 각 sample의 가까운 이웃만 연결한 kNN graph를 만든다. global dynamic clustering에서는 k 값을 여러 개 확인했다.

```text
k values checked: 30, 50, 100
main k: 50
```

즉 각 sample은 정규화된 dynamic feature space에서 가까운 이웃들과 연결된다. 이 graph는 “전체 공간에서 누가 누구와 가까운 dynamic behavior를 보이는가”를 표현한다.

## 10. Leiden clustering과 D-label

kNN graph를 만든 뒤, graph 위에서 Leiden clustering을 수행한다. Leiden clustering은 graph에서 연결이 촘촘한 community를 찾는 방법이다. 여기서는 raw coordinate space에서 둥근 cluster를 가정하는 k-means보다, kNN graph 위의 local neighborhood structure를 보는 방식에 가깝다.

global dynamic clustering의 핵심 흐름은 다음과 같다.

```text
Hamiltonian H
-> simulator trajectory at lambda=35
-> 472D dynamic feature x_dyn(H)
-> median/IQR normalization z_dyn(H)
-> L2-based approximate kNN graph
-> Leiden community detection
-> global dynamic family D000-D012
```

최종 label은 cluster 크기순으로 다시 붙인다. 따라서 `D000`은 가장 큰 dynamic family이고, `D012`는 가장 작은 dynamic family다. label 번호 자체가 물리적 순서를 뜻하는 것은 아니다.

## 11. S-family와 D/S 관계

일부 분석에서는 `D000`-`D012`와 함께 `S000`-`S037` structural mode도 사용한다. 두 label의 의미는 다르다.

```text
D-family: simulator trajectory 기반 dynamic-response label
S-family: Hamiltonian structural embedding 기반 structural mode label
```

`S-family`는 trajectory를 직접 사용하지 않고 Hamiltonian 구조 embedding을 기준으로 얻은 label이다. 따라서 같은 S-family에 속한다는 것은 Hamiltonian의 구조적 표현이 비슷하다는 뜻에 가깝고, 같은 D-family에 속한다는 것은 simulator에서 나온 transfer trajectory가 비슷하다는 뜻에 가깝다.

D/S 관계를 같이 보는 이유는 structural mode가 dynamic response를 얼마나 설명하는지 확인하기 위해서다. 만약 S와 D가 거의 1:1로 대응한다면, 구조적 family만으로 dynamic family를 지정할 수 있다. 하지만 실제 분석에서는 S-D 관계가 무작위 기준선보다는 높아도 강한 1:1 대응은 아니었다. 따라서 structural mode는 dynamic behavior를 일부 담고 있지만, dynamic-response diversity를 완전히 대체하지는 못한다.

보고서에서는 이를 다음처럼 해석하는 것이 안전하다.

```text
S-family는 dynamic family와 약하게 연결된 structural reference label이다.
D-family는 simulated trajectory가 비슷한 sample을 묶은 dynamic-response reference label이다.
둘의 관계는 structural feature만으로 dynamic response를 완전히 지정하기 어렵다는 근거로 사용한다.
```

## 12. D-family를 어떻게 해석해야 하는가

`D000`-`D012`는 Hamiltonian matrix의 structural cluster가 아니다. 같은 D-family에 속한다는 것은, 해당 Hamiltonian들이 lambda=35 simulation에서 비슷한 eta/path trajectory를 보였다는 뜻이다.

따라서 다음 표현은 안전하다.

```text
D-family는 lambda=35 조건에서 비슷한 energy-transfer trajectory를 보이는 dynamic behavior family다.
```

반대로 다음 표현은 조심해야 한다.

```text
D-family는 서로 다른 물리 mechanism을 증명한다.
D-family는 Hamiltonian 구조가 같은 cluster다.
D-family별 eta 차이는 완전히 독립적인 검증 결과다.
```

특히 eta trajectory와 eta summary metric이 clustering input에 포함되어 있으므로, D-family별 eta 차이는 어느 정도 clustering 기준 자체에 반영된 결과다. 따라서 D별 eta profile을 보여줄 때는 “D-family가 포착한 dynamic behavior 차이를 해석한다”라고 써야지, “D-family가 eta 차이를 독립적으로 예측했다”라고 쓰면 안 된다.

## 13. 사후 분석에 쓸 수 있는 값과 주의점

global D-family를 만든 feature에는 `eta_t`, `d eta/dt`, `path_t`, eta/timing/path summary가 들어간다. 반면 `c_l1`, `purity`, `ipr`은 직접 clustering input에 포함되지 않았다.

따라서 D-family별로 `c_l1`, `purity`, `ipr` 분포를 사후 비교하는 것은 비교적 의미가 있다. 예를 들어 특정 D-family가 coherence proxy나 localization proxy와 어떤 관계를 갖는지 보는 분석은 clustering input을 그대로 다시 확인하는 것보다는 더 독립적인 해석에 가깝다.

반대로 D-family별 `eta`, `tau`, `trap`, `loss`, `sink34`, `detour567` 차이는 clustering input과 직접 연결되어 있다. 이런 값들은 family의 성격을 설명하는 데 유용하지만, 독립 검증 지표처럼 해석하면 안 된다.

## 14. 보고서에서 사용할 수 있는 요약 문장

보고서에는 다음 정도로 요약하는 것이 안전하다.

> Global dynamic family D000-D012는 raw Hamiltonian parameter를 직접 clustering한 결과가 아니라, 각 Hamiltonian을 lambda=35에서 forward simulation하여 얻은 eta/path trajectory를 472차원 dynamic feature로 변환한 뒤, median/IQR scaling과 L2 기반 kNN graph, Leiden clustering을 적용해 얻은 dynamic behavior family다. 따라서 D-family는 Hamiltonian 구조 자체의 cluster라기보다, energy-transfer trajectory가 비슷한 sample들의 graph-based family로 해석해야 한다.

그리고 한계는 다음처럼 덧붙일 수 있다.

> 이 clustering은 mechanism을 직접 증명하지 않는다. eta trajectory와 path summary가 입력에 포함되어 있으므로, D-family별 eta/path 차이는 clustering 기준의 일부를 해석하는 것이다. 다만 `c_l1`, `purity`, `ipr`처럼 clustering input에 직접 포함되지 않은 quantum-state summary와 D-family의 관계를 사후 분석하면, 각 dynamic family가 coherence/localization 측면에서 어떤 차이를 보이는지 탐색할 수 있다.

## 15. 핵심 요약

global D000-D012 clustering은 다음 세 문장으로 요약할 수 있다.

1. 각 Hamiltonian을 lambda=35에서 simulator에 넣고, 0-50 ps trajectory를 얻었다.
2. eta trajectory, eta 변화율, path group trajectory, summary dynamic metrics를 합쳐 472차원 dynamic feature를 만들고 robust normalization했다.
3. 이 feature space에서 L2 기반 kNN graph를 만든 뒤 Leiden clustering을 적용해 D000-D012 dynamic family를 정의했다.

따라서 D-family는 “비슷한 Hamiltonian 구조”가 아니라 “비슷한 simulated transfer trajectory”를 기준으로 한 global dynamic family다.
