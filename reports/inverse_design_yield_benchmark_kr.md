# H27 Inverse-Design Yield Benchmark

## 목적

이 로컬 실험은 이미 exact simulator validation이 끝난 generated samples를 같은 생성/시뮬레이션 예산으로 비교한다.
질문은 NLL이 아니라 `target 조건을 만족하는 Hamiltonian 후보를 얼마나 얻었는가`이다.

## 입력

- assignments: dynamic-distance reference assignment table from the compact result bundle
- targets: `fast_high, late_high, very_fast`
- nominal budget: `512` generated samples per target
- 주의: 현재 로컬에는 `torch`/`qutip`이 없어 새 생성 또는 simulator rerun은 하지 않았다.

## Aggregate exact-high yield

| model | n_generated_or_validated | n_target_match | mean_target_match_rate | min_target_match_rate | valid_designs_per_512_mean | eta20_median_targetmatch_mean | t80_median_targetmatch_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNF baseline | 1535 | 905 | 0.5896 | 0.5832 | 301.8611 | 0.8684 | 13.6635 |
| CNF_WMODE | 1536 | 878 | 0.5716 | 0.5527 | 292.6667 | 0.8766 | 13.1015 |
| FLOW_HTBRANCHPINNTRAJ | 1530 | 1023 | 0.6687 | 0.5577 | 342.3991 | 0.8877 | 12.4318 |
| HTBAL_CNF_MIXPRIOR | 1536 | 990 | 0.6445 | 0.6133 | 330.0000 | 0.8769 | 13.2706 |

해석:

- `FLOW_HTBRANCHPINNTRAJ`는 평균 target-match yield가 가장 높다. 같은 생성 예산에서 simulator가 검증한 target-match 후보를 가장 많이 얻는다는 점에서 main inverse-design yield 후보로 둔다.
- `HTBAL_CNF_MIXPRIOR`는 CNF baseline 대비 mean target-match가 `5.50` percentage point 높고, CNF_WMODE 대비 `7.29` percentage point 높다.
- 다만 `HTBAL_CNF_MIXPRIOR`는 FLOW보다 late_high target과 density/support 관점에서 더 보수적인 장점이 있으므로, 버리는 모델이 아니라 yield-density-support trade-off를 해석하기 위한 중요한 비교 축으로 둔다.

## Per-target exact-high yield

| model | target | n_generated_or_validated | n_target_match | target_match_rate | valid_designs_per_512 | eta20_median_targetmatch | t80_median_targetmatch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNF baseline | fast_high | 511 | 298 | 0.5832 | 298.5832 | 0.9059 | 10.9841 |
| CNF baseline | late_high | 512 | 300 | 0.5859 | 300.0000 | 0.7621 | 22.8684 |
| CNF baseline | very_fast | 512 | 307 | 0.5996 | 307.0000 | 0.9372 | 7.1379 |
| CNF_WMODE | fast_high | 512 | 307 | 0.5996 | 307.0000 | 0.9133 | 10.8117 |
| CNF_WMODE | late_high | 512 | 288 | 0.5625 | 288.0000 | 0.7798 | 21.4316 |
| CNF_WMODE | very_fast | 512 | 283 | 0.5527 | 283.0000 | 0.9366 | 7.0611 |
| FLOW_HTBRANCHPINNTRAJ | fast_high | 508 | 371 | 0.7303 | 373.9213 | 0.9188 | 10.3625 |
| FLOW_HTBRANCHPINNTRAJ | late_high | 511 | 285 | 0.5577 | 285.5577 | 0.7999 | 20.0119 |
| FLOW_HTBRANCHPINNTRAJ | very_fast | 511 | 367 | 0.7182 | 367.7182 | 0.9444 | 6.9211 |
| HTBAL_CNF_MIXPRIOR | fast_high | 512 | 343 | 0.6699 | 343.0000 | 0.9117 | 11.1873 |
| HTBAL_CNF_MIXPRIOR | late_high | 512 | 333 | 0.6504 | 333.0000 | 0.7798 | 21.5573 |
| HTBAL_CNF_MIXPRIOR | very_fast | 512 | 314 | 0.6133 | 314.0000 | 0.9391 | 7.0671 |

## 성공 샘플의 reference-family 분산

| model | target | n_targetmatch_assigned | dynamic_family_count | top_dynamic_family_id | top_dynamic_family_fraction | nearest_reference_distance_median | nearest_reference_distance_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FLOW_HTBRANCHPINNTRAJ | fast_high | 371 | 4 | D009 | 0.6550 | 0.0237 | 0.0513 |
| FLOW_HTBRANCHPINNTRAJ | late_high | 285 | 6 | D003 | 0.3930 | 0.0263 | 0.0353 |
| FLOW_HTBRANCHPINNTRAJ | very_fast | 367 | 3 | D009 | 0.5531 | 0.0163 | 0.0215 |
| HTBAL_CNF_MIXPRIOR | fast_high | 343 | 4 | D009 | 0.6356 | 0.0227 | 0.0388 |
| HTBAL_CNF_MIXPRIOR | late_high | 333 | 6 | D003 | 0.3964 | 0.0263 | 0.0339 |
| HTBAL_CNF_MIXPRIOR | very_fast | 314 | 3 | D009 | 0.6146 | 0.0165 | 0.0219 |

## 성공 샘플끼리의 dynamic-distance cluster caveat

| model | target | n_target_match | target_match_rate | targetmatch_largest_existing_cluster_fraction | targetmatch_existing_cluster_entropy_norm | targetmatch_cluster_counts |
| --- | --- | --- | --- | --- | --- | --- |
| CNF baseline | fast_high | 298 | 0.5832 | 1.0000 | 0.0000 | 0:298 |
| CNF baseline | late_high | 300 | 0.5859 | 0.9033 | 0.4583 | 0:29;1:271 |
| CNF baseline | very_fast | 307 | 0.5996 | 1.0000 | 0.0000 | 1:307 |
| CNF_WMODE | fast_high | 307 | 0.5996 | 1.0000 | 0.0000 | 0:307 |
| CNF_WMODE | late_high | 288 | 0.5625 | 0.9410 | 0.3236 | 0:17;1:271 |
| CNF_WMODE | very_fast | 283 | 0.5527 | 1.0000 | 0.0000 | 1:283 |
| FLOW_HTBRANCHPINNTRAJ | fast_high | 371 | 0.7303 | 0.7251 | 0.8485 | 0:269;2:102 |
| FLOW_HTBRANCHPINNTRAJ | late_high | 285 | 0.5577 | 0.5544 | 0.9914 | 0:158;1:127 |
| FLOW_HTBRANCHPINNTRAJ | very_fast | 367 | 0.7182 | 1.0000 | 0.0000 | 1:367 |
| HTBAL_CNF_MIXPRIOR | fast_high | 343 | 0.6699 | 1.0000 | 0.0000 | 0:343 |
| HTBAL_CNF_MIXPRIOR | late_high | 333 | 0.6504 | 0.9640 | 0.2238 | 0:12;1:321 |
| HTBAL_CNF_MIXPRIOR | very_fast | 314 | 0.6133 | 1.0000 | 0.0000 | 1:314 |

## 결론

`FLOW_HTBRANCHPINNTRAJ`는 평균 exact-high inverse-design yield를 가장 크게 끌어올린다.
따라서 FLOW를 main inverse-design yield 후보로 두는 것이 현재 결과와 가장 잘 맞는다.
다만 FLOW가 모든 지표에서 우월하다는 뜻은 아니다. `HTBAL_CNF_MIXPRIOR`는 late_high target과 support/density 관점에서 더 보수적인 장점을 보인다.
따라서 최종 해석은 “FLOW가 main yield 후보이고, MIXPRIOR/HTBAL은 density/support/diversity trade-off를 보여주는 비교 모델”에 가깝다.

또한 이 실험이 보여주는 것은 `valid candidate generation 가능성`이지, H-space novelty나 robustness proof까지는 아니다.

다음 단계에서 더 강한 claim을 하려면 generated H NPZ와 checkpoint를 가져와서 `nearest train-H distance`, top-K ranking, lambda/noise robustness를 새로 계산해야 한다.
