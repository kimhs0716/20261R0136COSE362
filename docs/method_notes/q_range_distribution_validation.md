# Feasible 분포 검증 방향: q-range validation

이 문서는 생성된 Hamiltonian이 물리적으로 그럴듯한 범위 안에 있는지 검증하는 방향을 정리한 것이다. 핵심은 generated distribution이 `q(H)`와 같아야 한다는 뜻이 아니라, generated Hamiltonian이 `q(H)`가 정의하는 feasible range 밖으로 벗어나지 않아야 한다는 점이다.

## 1. 핵심 목표

우리가 원하는 것은 다음이 아니다.

$$
p_\theta(H \mid c) \approx q(H)
$$

이 조건은 너무 강하다. 특정 조건 $c$를 주면 generated Hamiltonian은 전체 prior $q(H)$가 아니라, 그 조건을 만족하는 부분집합에 몰리는 것이 자연스럽다.

따라서 실제 목표는 다음에 가깝다.

$$
\operatorname{supp}\bigl(p_\theta(H \mid c)\bigr)
\subseteq
\operatorname{supp}\bigl(q(H)\bigr)
$$

즉, 조건 $c$를 만족하는 Hamiltonian을 생성하되, geometry-based feasible prior가 만들 수 없는 영역으로 벗어나면 안 된다.

이를 sample 단위로 쓰면 다음과 같다.

$$
H_{\mathrm{gen}} \sim p_\theta(H \mid c),
\qquad
H_{\mathrm{gen}} \in R_q,
\qquad
f_{\mathrm{sim}}(H_{\mathrm{gen}}) \approx c
$$

여기서 $R_q$는 `sample_H_geom`이 유도하는 feasible Hamiltonian range이고, $f_{\mathrm{sim}}$은 Hamiltonian을 시뮬레이션해서 condition label을 계산하는 simulator다.

## 2. q(H)의 정의

본 프로젝트에서 $q(H)$는 자연계의 모든 가능한 FMO Hamiltonian 분포가 아니다. `sample_H_geom`이 유도하는 geometry-based feasible Hamiltonian prior다.

먼저 latent geometry variable을 다음처럼 둔다.

$$
z =
(r_1,\ldots,r_7,\,
\mu_1,\ldots,\mu_7,\,
e_1,\ldots,e_7)
$$

각 변수는 다음 조건을 따른다.

$$
r_i \in [0,4]^3,
\qquad
\lVert r_i-r_j\rVert_2 \ge 1,
\qquad
\lVert \mu_i\rVert_2 = 1,
\qquad
e_i \in [0,450]
$$

`sample_H_geom`은 이 latent variable $z$를 Hamiltonian으로 보내는 deterministic map으로 볼 수 있다.

$$
H = S(z)
$$

Off-diagonal coupling은 다음 형태로 계산된다.

$$
H_{ij}
=
\frac{150\,\kappa_{ij}}{\lVert r_i-r_j\rVert_2^3},
\qquad i\ne j
$$

여기서 orientation factor는 다음과 같다.

$$
\kappa_{ij}
=
\mu_i^\top \mu_j
-3(\mu_i^\top \hat r_{ij})(\mu_j^\top \hat r_{ij}),
\qquad
\hat r_{ij}
=
\frac{r_i-r_j}{\lVert r_i-r_j\rVert_2}
$$

Diagonal term은 평균을 제거해 trace-zero gauge로 저장한다.

$$
H_{ii}=e_i-\bar e,
\qquad
\bar e=\frac{1}{7}\sum_{k=1}^{7}e_k
$$

따라서 $q(H)$는 latent distribution $p(z)$가 $S$를 통해 유도하는 push-forward distribution이다.

$$
q(H)=S_{\#}p(z)
$$

Feasible range는 다음처럼 정의한다.

$$
R_q=\{S(z): z\in Z\}
$$

따라서 이 문서에서 feasible하다는 말은 다음을 의미한다.

$$
H \in R_q
$$

이는 “자연계에서 실제로 가능한 모든 FMO Hamiltonian”이라는 뜻이 아니라, “본 프로젝트의 geometry-based sampling assumption 아래에서 가능한 Hamiltonian”이라는 뜻이다.

## 3. 검증을 두 층으로 나누기

### 3.1 Sample-level q-range membership

각 generated Hamiltonian에 대해 다음을 묻는다.

$$
H_{\mathrm{gen}}\in R_q \; ?
$$

즉, 개별 sample 하나가 geometry-based feasible range 안에 들어오는지를 보는 검증이다.

### 3.2 Distribution-level leakage rate

생성 분포 전체에 대해서는 q-range 밖으로 새는 비율을 본다.

$$
V_q(c)
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbb{1}
\left[
H_m \notin R_q
\right],
\qquad
H_m \sim p_\theta(H \mid c)
$$

$V_q(c)$가 낮을수록, 조건 $c$에서 생성된 Hamiltonian들이 feasible range 안에 잘 머문다고 볼 수 있다.

중요한 점은 $V_q(c)$가 $p_\theta(H \mid c)$와 $q(H)$의 분포 동일성을 보는 지표가 아니라는 것이다. 이 값은 generated distribution이 $q(H)$의 support 밖으로 얼마나 벗어나는지를 보는 leakage 지표다.

## 4. q-range 판정 방법

$q(H)$는 closed-form density가 없고, $R_q$도 간단한 부등식 몇 개로 완전히 표현하기 어렵다. 따라서 q-range membership은 여러 detector를 함께 사용해 근사적으로 판단한다.

### 4.1 Hard necessary checks

가장 먼저 확실한 필요조건을 확인한다.

$$
H=H^\top
$$

$$
\operatorname{Tr}(H)\approx 0
$$

$$
\max_i H_{ii}-\min_i H_{ii}\le 450
$$

$$
\max_{i<j}|H_{ij}|\le 300
$$

마지막 조건은 다음 사실에서 나온다.

$$
|H_{ij}|
=
\left|
\frac{150\,\kappa_{ij}}{r_{ij}^3}
\right|,
\qquad
|\kappa_{ij}|\le 2,
\qquad
r_{ij}\ge 1
$$

따라서

$$
|H_{ij}|\le 300
$$

이 조건을 깨면 q-range 밖이라고 볼 수 있다. 다만 이 조건들은 necessary condition이지 sufficient condition은 아니다. 즉, 통과했다고 해서 반드시 $H\in R_q$라고 단정할 수는 없다.

### 4.2 q vs non-q support classifier

가장 실용적인 방법은 q-range membership classifier를 학습하는 것이다.

학습 데이터는 다음처럼 구성한다.

$$
y=1:\; H\sim q(H)
$$

$$
y=0:\; H\sim q_{\mathrm{neg}}(H)
$$

여기서 $q_{\mathrm{neg}}(H)$는 q-range 밖의 Hamiltonian을 의도적으로 만든 negative distribution이다. 예를 들면 다음과 같다.

- independent uniform off-diagonal Hamiltonian
- off-diagonal shuffled Hamiltonian
- 특정 coupling edge만 비정상적으로 키운 Hamiltonian
- diagonal range를 크게 벗어난 Hamiltonian
- random symmetric trace-zero Hamiltonian
- q sample에 large perturbation을 더한 Hamiltonian
- coupling magnitude는 유지하되 pair assignment를 섞은 Hamiltonian
- row 또는 degree structure를 깨는 perturbation Hamiltonian

Classifier는 다음 값을 근사한다.

$$
D_\psi(H)\approx P(H\in R_q)
$$

Generated Hamiltonian에 대해

$$
D_\psi(H_{\mathrm{gen}})
$$

이 낮으면 q-range 밖일 가능성이 크다고 판단한다.

### 4.3 Autoencoder support score

q sample만 이용해 autoencoder 또는 denoising autoencoder를 학습할 수도 있다.

$$
H \rightarrow z \rightarrow \hat H
$$

Support score는 reconstruction error로 정의한다.

$$
S_{\mathrm{AE}}(H)
=
\lVert H-\hat H\rVert_2^2
$$

q-range 근처의 Hamiltonian은 reconstruction error가 작고, q-range 밖의 Hamiltonian은 reconstruction error가 커질 것으로 기대한다.

### 4.4 Unconditional q-flow

별도의 unconditional flow로 $q(H)$를 근사할 수도 있다.

$$
q_\phi(H)\approx q(H)
$$

그 후 generated sample에 대해 다음 값을 본다.

$$
\log q_\phi(H_{\mathrm{gen}})
$$

다만 이 값은 support membership이라기보다 density 또는 OOD score에 가깝다. Low likelihood라고 해서 반드시 q-range 밖이라는 뜻은 아니다. q-range 안에 있지만 rare한 Hamiltonian일 수도 있다.

따라서 q-flow likelihood는 보조 지표로 쓰는 것이 안전하다.

### 4.5 Projection-based confirmation

가장 직접적인 q-range distance는 다음과 같이 정의할 수 있다.

$$
d_q(H)
=
\min_{z\in Z}
\lVert H-S(z)\rVert_2^2
$$

$d_q(H)$가 0에 가까우면 q-range 안에 가까운 Hamiltonian이라고 볼 수 있다.

다만 이 최적화는 계산량이 크고 non-convex일 수 있다. 따라서 모든 sample에 적용하기보다는, 대표 sample이나 q-range violation이 의심되는 sample에 대한 case study로 사용하는 것이 적절하다.

## 5. q-range 안에 머물게 하는 방법

### 5.1 Post-filtering

현재 H-space conditional generator를 유지한다면 가장 현실적인 방법은 post-filtering이다.

먼저 많이 생성한다.

$$
H_m\sim p_\theta(H\mid c),
\qquad m=1,\ldots,M
$$

그 뒤 q-range detector와 simulator 재검증을 모두 통과한 sample만 남긴다.

$$
D_\psi(H_m)>\tau_q
$$

$$
f_{\mathrm{sim}}(H_m)\approx c
$$

즉 최종 accept 조건은 다음과 같다.

$$
D_\psi(H_m)>\tau_q
\quad\text{and}\quad
f_{\mathrm{sim}}(H_m)\approx c
$$

이 방식은 꼼수가 아니라, implicit prior가 있는 inverse design 문제에서 자연스러운 feasibility screen으로 볼 수 있다.

### 5.2 q-score 기반 ranking

Post-filtering보다 부드러운 방법은 q-score 기반 ranking이다.

$$
\operatorname{Score}(H)
=
\alpha\,\operatorname{ConditionScore}(H,c)
+\beta\,\operatorname{QSupportScore}(H)
+\gamma\,\log p_\theta(H\mid c)
$$

Classifier를 쓰면 q-support score를 다음처럼 둘 수 있다.

$$
\operatorname{QSupportScore}(H)=\log D_\psi(H)
$$

Autoencoder를 쓰면 다음처럼 둘 수 있다.

$$
\operatorname{QSupportScore}(H)=-S_{\mathrm{AE}}(H)
$$

이 방식은 condition만 좋은 Hamiltonian이 아니라, condition을 만족하면서 q-range 안에 있을 가능성이 높은 Hamiltonian을 우선 선택하는 데 적합하다.

### 5.3 Training-time q-regularization

후속 단계에서는 q-support classifier나 AE score를 regularization으로 사용할 수도 있다.

Classifier 기반 penalty는 다음처럼 쓸 수 있다.

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{gen}}
+\lambda
\mathbb{E}_{H\sim p_\theta(H\mid c)}
\left[
\operatorname{ReLU}(\tau_q-D_\psi(H))^2
\right]
$$

AE 기반 penalty는 다음처럼 쓸 수 있다.

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{gen}}
+\lambda
\mathbb{E}_{H\sim p_\theta(H\mid c)}
\left[
\operatorname{ReLU}(S_{\mathrm{AE}}(H)-\tau_{\mathrm{AE}})^2
\right]
$$

다만 현재 단계에서는 구현 복잡도를 고려해 post-filtering과 q-score ranking을 우선 적용하는 것이 현실적이다.

## 6. 최종 검증 pipeline

현실적인 검증 순서는 다음과 같다.

1. `sample_H_geom`으로 q-reference set을 만든다.
2. 여러 방식으로 q-negative set을 만든다.
3. q vs non-q support classifier를 학습한다.
4. q-validation set에서 threshold $\tau_q$를 정한다.
5. 조건 $c$에서 generated Hamiltonian을 많이 만든다.
6. 각 sample에 대해 hard check, q-classifier score, optional AE score, simulator condition check를 계산한다.
7. 최종적으로 q-range pass rate, condition pass rate, joint pass rate를 보고한다.

q-range pass rate는 다음과 같다.

$$
\operatorname{PassRate}_q(c)
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbb{1}
\left[
D_\psi(H_m)>\tau_q
\right]
$$

Condition pass rate는 다음과 같다.

$$
\operatorname{PassRate}_c(c)
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbb{1}
\left[
f_{\mathrm{sim}}(H_m)\approx c
\right]
$$

가장 중요한 지표는 joint pass rate다.

$$
\operatorname{PassRate}_{q,c}(c)
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbb{1}
\left[
D_\psi(H_m)>\tau_q
\;\land\;
f_{\mathrm{sim}}(H_m)\approx c
\right]
$$

이 값은 generated Hamiltonian이 feasible range 안에 있으면서 target condition도 만족하는 비율을 의미한다.

## 7. Coverage check의 위치

Coverage는 q-range validation 이후에 보는 보조 분석으로 두는 것이 좋다.

우선순위는 다음과 같다.

1. generated Hamiltonian이 q-range 밖으로 나가지 않는가?
2. generated Hamiltonian이 target condition을 만족하는가?
3. q-range 안에서 다양한 영역을 덮는가?

즉 먼저 q-range validity와 condition validity를 확인하고, 그 다음에 mode collapse 여부나 cluster coverage entropy를 본다.

Coverage는 예를 들어 다음처럼 볼 수 있다.

$$
H_m \mapsto k_m
$$

여기서 $k_m$은 q-reference clustering에서 assign된 cluster index다. 생성 샘플의 cluster distribution을 $P_{\mathrm{gen}}(k\mid c)$라고 하면, coverage entropy는 다음처럼 쓸 수 있다.

$$
\mathcal{H}_{\mathrm{coverage}}(c)
=
-
\sum_k
P_{\mathrm{gen}}(k\mid c)
\log P_{\mathrm{gen}}(k\mid c)
$$

이 entropy가 너무 낮으면 q-range 안에 있더라도 일부 mode에만 몰리는 mode collapse 가능성이 있다.

## 8. 보고서용 핵심 문장

보고서에서는 다음처럼 정리할 수 있다.

본 연구에서 feasible Hamiltonian은 `sample_H_geom`이 유도하는 geometry-based prior $q(H)$의 range 안에 있는 Hamiltonian으로 정의한다. 여기서 $q(H)$는 자연계에서 가능한 모든 FMO Hamiltonian의 분포가 아니라, 본 연구의 sampling assumption 하에서 가능한 Hamiltonian prior이다. 따라서 조건부 생성 모델의 목표는 단순히 scalar condition $c$를 만족하는 임의의 matrix를 생성하는 것이 아니라, $q(H)$의 feasible range 안에 머물면서 $c$를 만족하는 Hamiltonian을 생성하는 것이다. 이를 위해 생성된 Hamiltonian에 대해 q-range membership과 simulator-based condition validity를 함께 평가한다.
