# Hamiltonian 샘플링 방법: `sample_H_geom`

이 문서는 초기 데이터셋에서 Hamiltonian `H`를 어떻게 샘플링했는지 정리한 것이다. 핵심은 `sample_H_geom` 함수이며, 이후 시뮬레이션으로 label을 붙이는 과정은 별도 단계로 본다.

## 1. 목적

초기에는 7x7 Hamiltonian의 off-diagonal coupling을 단순히 균등분포에서 독립적으로 뽑는 방식도 고려할 수 있었다. 하지만 그런 방식은 모든 site pair가 서로 비슷한 확률로 강하게 연결될 수 있어, 실제 3차원 배치에서 나오는 Hamiltonian이라고 보기 어렵다.

그래서 `sample_H_geom`에서는 7개의 site를 3차원 공간에 배치하고, site 사이 거리와 dipole orientation을 이용해 coupling을 계산한다. 즉, Hamiltonian을 직접 임의로 뽑는 것이 아니라, 간단한 기하 구조를 먼저 만들고 그로부터 Hamiltonian을 구성한다.

## 2. 간단한 이론 배경

FMO complex는 여러 개의 pigment site 사이에서 excitation energy가 이동하는 시스템으로 볼 수 있다. 이때 site basis에서 Hamiltonian의 diagonal term은 각 site의 excitation energy를 나타내고, off-diagonal term은 서로 다른 site 사이의 excitonic coupling을 나타낸다.

```text
H_ii: site i의 excitation energy
H_ij: site i와 site j 사이의 coupling
```

두 pigment 사이의 coupling은 실제로는 복잡한 전자구조와 주변 환경의 영향을 받지만, 단순화된 모델에서는 transition dipole 사이의 dipole-dipole interaction으로 근사할 수 있다. 이 근사에서는 두 site 사이의 거리가 가까울수록 coupling이 커지고, dipole 방향이 어떻게 놓여 있는지에 따라 coupling의 부호와 크기가 달라진다.

이때 coupling의 기본 형태는 다음과 같다.

```text
V_ij proportional to kappa_ij / r_ij^3
```

여기서 `r_ij`는 두 site 사이 거리이고, `kappa_ij`는 두 dipole의 상대적 방향을 나타내는 orientation factor다. 따라서 `sample_H_geom`의 핵심 가정은 “Hamiltonian의 coupling 구조는 완전히 독립적인 난수가 아니라, 3차원 거리와 dipole 방향에서 유도된다”는 것이다.

이 방식은 실제 FMO 구조를 정밀하게 재현하는 ab initio 모델은 아니지만, 최소한 다음의 물리적 직관을 반영한다.

- 멀리 떨어진 site pair는 강하게 coupling되기 어렵다.
- 가까운 site pair라도 dipole 방향에 따라 coupling이 약하거나 부호가 바뀔 수 있다.
- 모든 coupling을 독립적으로 균등 샘플링하는 것보다 geometry-consistent한 Hamiltonian prior를 만든다.

## 3. Site 위치 샘플링

7개의 site를 한 변의 길이가 4 nm인 정육면체 내부에 배치한다.

각 site의 좌표는 다음 범위에서 무작위로 뽑는다.

```text
x, y, z in [0, 4] nm
```

단, 두 site가 너무 가까워지는 것을 막기 위해 최소 거리 조건을 둔다.

```text
distance(site_i, site_j) > 1 nm
```

새로운 site를 뽑았을 때 기존 site들과의 거리가 1 nm보다 작으면 버리고 다시 뽑는다. 이 조건은 분자들이 같은 공간에 겹치지 않도록 하는 excluded-volume 가정으로 볼 수 있다.

## 4. Dipole 방향 샘플링

각 site에는 transition dipole 방향을 하나씩 부여한다.

구현에서는 3차원 정규분포에서 벡터를 뽑은 뒤, 길이가 1이 되도록 정규화한다.

```text
mu_i = random 3D vector / norm(random 3D vector)
```

따라서 각 site의 dipole은 무작위 방향을 가지는 unit vector가 된다.

## 5. Off-diagonal coupling 계산

서로 다른 두 site `i`, `j` 사이의 coupling은 point-dipole 형태로 계산한다.

두 site 사이 거리 벡터를 `r_ij`, 거리의 크기를 `r`, 단위 방향 벡터를 `r_hat`이라고 하면 orientation factor `kappa`는 다음과 같다.

```text
kappa_ij = mu_i dot mu_j - 3 * (mu_i dot r_hat) * (mu_j dot r_hat)
```

그리고 coupling은 다음과 같이 계산한다.

```text
V_ij = GEOM_DIP * kappa_ij / r^3
```

현재 코드에서 사용한 상수는 다음과 같다.

```text
GEOM_DIP = 150 cm^-1 nm^3
```

이렇게 하면 가까운 site일수록 coupling이 커질 수 있고, dipole 방향에 따라 coupling의 부호와 크기가 달라진다. 계산된 값은 대칭 행렬이 되도록 `H[i, j]`와 `H[j, i]`에 함께 넣는다.

## 6. Diagonal energy 샘플링

Diagonal term은 각 site의 site energy에 해당한다.

먼저 7개의 site energy를 다음 범위에서 무작위로 뽑는다.

```text
E_i in [0, 450] cm^-1
```

그 다음 전체 평균을 빼서 diagonal sum이 0이 되도록 맞춘다.

```text
H_ii = E_i - mean(E)
```

이 과정은 Hamiltonian의 공통 energy offset을 제거하는 gauge fixing으로 볼 수 있다. 실제 dynamics에서는 모든 diagonal에 같은 상수를 더해도 상대적인 에너지 차이와 coupling 구조가 핵심이므로, 평균을 빼서 trace-zero 형태로 저장한다.

## 7. 최종 출력

최종적으로 `sample_H_geom`은 다음 형태의 Hamiltonian을 반환한다.

```text
H: 7 x 7 real symmetric matrix, unit = cm^-1
```

구성은 다음과 같다.

- diagonal: 무작위 site energy를 평균 제거한 값
- off-diagonal: 3D 위치와 dipole 방향으로 계산한 point-dipole coupling

옵션으로 `return_meta=True`를 사용하면 Hamiltonian뿐 아니라 다음 정보도 함께 받을 수 있다.

- `geom_pos`: 각 site의 3D 좌표
- `dipole_mu`: 각 site의 unit dipole vector
- `dist_mat`: site pair 사이 거리 행렬
- `kappa_mat`: site pair 사이 orientation factor 행렬

## 8. 해석상 주의점

이 샘플링 방식은 실제 FMO complex의 정확한 구조를 복원하는 것이 아니다. 4 nm box, 1 nm minimum distance, random dipole orientation, point-dipole coupling이라는 단순화된 가정을 사용한다.

따라서 이 방식은 “실제 FMO 구조를 그대로 샘플링했다”기보다는, “3차원 배치와 dipole coupling이라는 물리적으로 그럴듯한 제약을 반영해 Hamiltonian 후보를 만든다” 정도로 해석하는 것이 적절하다.

또한 site 번호는 샘플링 과정에서 특별한 공간적 의미를 갖지 않는다. 예를 들어 site 3과 site 4가 항상 가깝다거나, site 5-7이 특정 detour path를 이룬다는 조건은 `sample_H_geom` 자체에는 들어 있지 않다.

## 9. 전체 데이터 생성에서의 위치

`sample_H_geom`은 Hamiltonian 후보 `H`를 생성하는 단계다. 이후 각 `H`에 대해 simulator를 실행하여 `eta`, `tau_transfer`, `ipr`, `purity`, `c_l1`, time-dependent population 등의 label을 계산한다.

즉 전체 데이터 생성 흐름은 다음과 같다.

```text
3D site 위치 샘플링
-> dipole 방향 샘플링
-> point-dipole coupling 계산
-> diagonal energy 샘플링 및 trace-zero gauge fixing
-> 7x7 Hamiltonian H 생성
-> simulator로 label 계산
```
