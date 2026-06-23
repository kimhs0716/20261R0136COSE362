# H27 c_l1-free surrogate CNF Colab 실행 메모

노트북:

- `notebooks/h27_c_l1_free_surrogate_cnf_colab.ipynb`

스크립트:

- `scripts/train_h27_c_l1_free_surrogate_cnf.py`

## 목적

이 실험은 기존 CNF/mode-prior 계열을 다시 돌리되, 내부 dynamic mode / surrogate label을 `c_l1` 없는 dynamic distance 기준으로 다시 만든다.

사용하는 기준 feature:

- `eta_t`
- `path_t`
- `pop_t[:, :7]`

사용하지 않는 것:

- `c_l1`
- `dyn_z` PCA shortcut
- static `H` L2
- generated sample끼리만 만든 k-means cluster

## 실행 모델

기본 notebook 설정은 아래 세 모델을 비교한다.

- `CFAST_ORANGE3_CNF`: branch/mode 없는 clean control
- `CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR`: corrected reference family prior를 쓰는 internal mode-prior CNF
- `CFAST_ORANGE3_HTBAL_CNF_GUIDED`: corrected surrogate를 학습하고, 생성 시 latent refinement에 쓰는 guided CNF

## 중요한 차이

이전 audit notebook은 "무엇이 잘못됐는지" 확인하는 노트북이었다. 이 notebook은 그 결론을 반영해서 실제 corrected surrogate/CNF를 새로 돌리는 노트북이다.

기존 `dynamic_label`은 `CFAST + c_l1 + PATH`에서 만들어진 label일 수 있었으므로 diversity auxiliary로 쓰기 위험했다. 새 wrapper는 62k scalable reference의 `dynamic_family_id`를 `eta/path/pop` distance feature prototype으로 바꿔 140k prepared row에 다시 붙인다.

## 결과 판정

학습 뒤 notebook은 다음을 자동으로 실행한다.

1. generated H 샘플 저장
2. forward simulator validation
3. trajectory 저장
4. `assign_h27_generated_to_scalable_dynamic_reference.py`로 c_l1-free reference family assignment
5. `top_dynamic_family_fraction`, `dynamic_family_entropy_norm`, `dynamic_family_count` 확인

최종 diversity 판정은 training loss가 아니라 simulator validation 후 scalable reference assignment 결과로 한다.

## Reference feature cache

Drive에는 큰 단일 cache 파일 `scalable_dynamic_reference_features_t101_eta_path_pop.npz` 대신 다음 sharded cache가 있을 수 있다.

- `scalable_dynamic_reference_features_t101_eta_path_pop_manifest.json`
- `scalable_dynamic_reference_features_t101_eta_path_pop_shard00.npz`
- `scalable_dynamic_reference_features_t101_eta_path_pop_shard01.npz`
- `scalable_dynamic_reference_features_t101_eta_path_pop_shard02.npz`
- `scalable_dynamic_reference_features_t101_eta_path_pop_shard03.npz`

Notebook은 단일 `.npz`가 없으면 manifest를 자동으로 사용한다. 따라서 `missing reference feature cache: ...eta_path_pop.npz` 오류가 나면 notebook/script가 오래된 버전이다.
