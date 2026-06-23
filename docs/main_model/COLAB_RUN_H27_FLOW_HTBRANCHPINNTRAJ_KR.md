# H27 FLOW_HTBRANCHPINNTRAJ Colab 실행 메모

## 목적

이 실험은 diffusion이 아니라 conditional RealNVP/DNF-style flow다. 다만 이전 `h27_branch_diverse_flow_colab.ipynb`처럼 branch one-hot을 condition vector에 붙이지 않는다.

핵심 차이:

- user-facing condition은 `CFAST_ORANGE3` compact condition으로 유지한다.
- dynamic branch id는 내부 latent로만 쓴다.
- RealNVP coupling network 입력에는 `branch_embedding(internal_branch_id)`가 실제로 들어간다.
- surrogate는 diffusion PINNTRAJ 계열과 같은 `H -> population trajectory` PINN surrogate를 쓴다.
- 이전 flow의 branch-classifier surrogate는 쓰지 않는다.

따라서 이 실험은 flow 기반의 `p_theta(H | compact condition, internal branch)` ablation이다.

## 실행 파일

- Script: `scripts/train_h27_flow_htbranch_pinntraj.py`
- Notebook: `notebooks/h27_flow_htbranchpinntraj_colab.ipynb`

## 기본 출력

```text
outputs/experiments/20260622_h27_flow_htbranchpinntraj/
```

주요 산출물:

- `metadata/flow_htbranch_manifest.json`
- `metadata/flow_htbranch_internal_branch_summary.csv`
- `metadata/flow_htbranch_row_assignments.csv`
- `full/CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ_generated_samples.npz`
- `full/CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ_generation_provenance.csv`

## 비교 기준

이 실험은 이전 `CFAST_CL1_PATH_DYNZ_HYBRIDBRANCH`와 다음 차이를 확인하기 위한 것이다.

- branch one-hot condition 방식보다 internal branch embedding이 나은가
- H-space diversity뿐 아니라 simulator dynamic-feature diversity가 살아나는가
- high-transfer target에서 largest dynamic cluster fraction이 줄어드는가
- target match rate가 크게 망가지지 않는가

최종 판단은 반드시 simulator validation과 generated diversity audit으로 한다.

## Surrogate 차이

Diffusion HTBRANCH와 이 flow HTBRANCH는 같은 계열의 PINN trajectory surrogate를 쓴다.

이전 branch-diverse flow는 별도 branch surrogate를 썼다. 그 surrogate는 `H -> trajectory + branch classifier` 구조였고, branch CE/prototype loss를 제공했다. 이번 실험은 branch classifier surrogate를 쓰지 않고, branch는 generator 내부 latent/embedding으로 처리한다.
