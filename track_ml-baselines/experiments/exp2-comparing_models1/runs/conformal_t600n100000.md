# conformal + selective prediction (alpha = 0.1)

## Split conformal — calibrated on val, applied to test (90% target)

| split | arm | seeds | half-width | coverage | test RMSE |
|---|---|---|---|---|---|
| cold_drug | flaml-ecfp | 3 | 1.543 ± 0.020 | 0.898 ± 0.002 | 0.948 |
| cold_drug | flaml-chemberta | 3 | 1.705 ± 0.037 | 0.900 ± 0.003 | 1.041 |
| cold_drug | siamese-ecfp | 3 | 1.780 ± 0.006 | 0.900 ± 0.002 | 1.083 |
| cold_drug | siamese | 3 | 1.760 ± 0.009 | 0.902 ± 0.004 | 1.060 |
| cold_drug | fusion | 3 | 1.720 ± 0.014 | 0.899 ± 0.002 | 1.051 |
| scaffold | flaml-ecfp | 3 | 1.640 ± 0.033 | 0.903 ± 0.005 | 0.995 |
| scaffold | flaml-chemberta | 3 | 1.761 ± 0.014 | 0.899 ± 0.002 | 1.082 |
| scaffold | siamese-ecfp | 3 | 1.874 ± 0.021 | 0.899 ± 0.003 | 1.145 |
| scaffold | siamese | 3 | 1.819 ± 0.010 | 0.899 ± 0.004 | 1.112 |
| scaffold | fusion | 3 | 1.799 ± 0.034 | 0.897 ± 0.006 | 1.101 |
| cold_target | flaml-ecfp | 3 | 2.144 ± 0.026 | 0.886 ± 0.013 | 1.384 |
| cold_target | flaml-chemberta | 3 | 2.088 ± 0.034 | 0.874 ± 0.016 | 1.376 |
| cold_target | siamese-ecfp | 3 | 2.233 ± 0.023 | 0.882 ± 0.015 | 1.437 |
| cold_target | siamese | 3 | 2.127 ± 0.035 | 0.871 ± 0.018 | 1.413 |
| cold_target | fusion | 3 | 2.101 ± 0.032 | 0.870 ± 0.023 | 1.400 |
| cold_both | flaml-ecfp | 3 | 2.133 ± 0.075 | 0.888 ± 0.007 | 1.361 |
| cold_both | flaml-chemberta | 3 | 2.056 ± 0.096 | 0.887 ± 0.013 | 1.327 |
| cold_both | siamese-ecfp | 3 | 2.271 ± 0.107 | 0.886 ± 0.011 | 1.403 |
| cold_both | siamese | 3 | 2.153 ± 0.109 | 0.893 ± 0.014 | 1.339 |
| cold_both | fusion | 3 | 2.169 ± 0.090 | 0.888 ± 0.026 | 1.374 |

## Selective prediction — RMSE on what is kept

`spread` ranks by how much the arms disagree on that pair; `oracle` ranks by the
true error (the ceiling); `random` abstains blindly (the floor).

| split | arm | score | @0% | @10% | @20% | @30% |
|---|---|---|---|---|---|---|

### How much of the achievable gain the disagreement score captures

`(rmse@0 - spread@r) / (rmse@0 - oracle@r)`, averaged over arms. 1.0 would mean
the score ranks pairs as well as knowing the true error; 0.0 means it is no
better than abstaining at random.

| split | @10% | @20% | @30% |
|---|---|---|---|
| cold_drug | 23% | 23% | 25% |
| scaffold | 21% | 22% | 24% |
| cold_target | 6% | 7% | 7% |
| cold_both | 2% | 4% | 7% |
| cold_drug | flaml-ecfp | oracle | 0.948 | 0.696 | 0.566 | 0.465 |
| cold_drug | flaml-ecfp | random | 0.948 | 0.943 | 0.945 | 0.948 |
| cold_drug | flaml-ecfp | spread | 0.948 | 0.912 | 0.889 | 0.860 |
| cold_drug | flaml-chemberta | oracle | 1.041 | 0.789 | 0.652 | 0.543 |
| cold_drug | flaml-chemberta | random | 1.041 | 1.039 | 1.037 | 1.041 |
| cold_drug | flaml-chemberta | spread | 1.041 | 0.987 | 0.953 | 0.917 |
| cold_drug | siamese-ecfp | oracle | 1.083 | 0.801 | 0.654 | 0.539 |
| cold_drug | siamese-ecfp | random | 1.083 | 1.081 | 1.083 | 1.087 |
| cold_drug | siamese-ecfp | spread | 1.083 | 1.006 | 0.963 | 0.922 |
| cold_drug | siamese | oracle | 1.060 | 0.775 | 0.628 | 0.516 |
| cold_drug | siamese | random | 1.060 | 1.060 | 1.060 | 1.061 |
| cold_drug | siamese | spread | 1.060 | 0.987 | 0.948 | 0.911 |
| cold_drug | fusion | oracle | 1.051 | 0.767 | 0.622 | 0.511 |
| cold_drug | fusion | random | 1.051 | 1.050 | 1.050 | 1.047 |
| cold_drug | fusion | spread | 1.051 | 0.981 | 0.943 | 0.906 |
| scaffold | flaml-ecfp | oracle | 0.995 | 0.732 | 0.598 | 0.492 |
| scaffold | flaml-ecfp | random | 0.995 | 0.994 | 0.993 | 0.993 |
| scaffold | flaml-ecfp | spread | 0.995 | 0.963 | 0.938 | 0.912 |
| scaffold | flaml-chemberta | oracle | 1.082 | 0.813 | 0.670 | 0.556 |
| scaffold | flaml-chemberta | random | 1.082 | 1.081 | 1.082 | 1.082 |
| scaffold | flaml-chemberta | spread | 1.082 | 1.032 | 0.998 | 0.963 |
| scaffold | siamese-ecfp | oracle | 1.145 | 0.847 | 0.692 | 0.570 |
| scaffold | siamese-ecfp | random | 1.145 | 1.144 | 1.144 | 1.145 |
| scaffold | siamese-ecfp | spread | 1.145 | 1.061 | 1.015 | 0.978 |
| scaffold | siamese | oracle | 1.112 | 0.823 | 0.671 | 0.554 |
| scaffold | siamese | random | 1.112 | 1.112 | 1.113 | 1.112 |
| scaffold | siamese | spread | 1.112 | 1.044 | 1.002 | 0.967 |
| scaffold | fusion | oracle | 1.101 | 0.806 | 0.654 | 0.537 |
| scaffold | fusion | random | 1.101 | 1.102 | 1.102 | 1.105 |
| scaffold | fusion | spread | 1.101 | 1.035 | 0.995 | 0.956 |
| cold_target | flaml-ecfp | oracle | 1.384 | 1.108 | 0.943 | 0.804 |
| cold_target | flaml-ecfp | random | 1.384 | 1.385 | 1.387 | 1.388 |
| cold_target | flaml-ecfp | spread | 1.384 | 1.375 | 1.364 | 1.354 |
| cold_target | flaml-chemberta | oracle | 1.376 | 1.105 | 0.940 | 0.802 |
| cold_target | flaml-chemberta | random | 1.376 | 1.376 | 1.379 | 1.378 |
| cold_target | flaml-chemberta | spread | 1.376 | 1.361 | 1.350 | 1.340 |
| cold_target | siamese-ecfp | oracle | 1.437 | 1.140 | 0.962 | 0.814 |
| cold_target | siamese-ecfp | random | 1.437 | 1.434 | 1.434 | 1.432 |
| cold_target | siamese-ecfp | spread | 1.437 | 1.410 | 1.391 | 1.374 |
| cold_target | siamese | oracle | 1.413 | 1.123 | 0.948 | 0.801 |
| cold_target | siamese | random | 1.413 | 1.410 | 1.410 | 1.408 |
| cold_target | siamese | spread | 1.413 | 1.389 | 1.376 | 1.361 |
| cold_target | fusion | oracle | 1.400 | 1.125 | 0.959 | 0.816 |
| cold_target | fusion | random | 1.400 | 1.404 | 1.399 | 1.398 |
| cold_target | fusion | spread | 1.400 | 1.385 | 1.375 | 1.362 |
| cold_both | flaml-ecfp | oracle | 1.361 | 1.105 | 0.943 | 0.808 |
| cold_both | flaml-ecfp | random | 1.361 | 1.368 | 1.357 | 1.336 |
| cold_both | flaml-ecfp | spread | 1.361 | 1.364 | 1.356 | 1.334 |
| cold_both | flaml-chemberta | oracle | 1.327 | 1.068 | 0.909 | 0.770 |
| cold_both | flaml-chemberta | random | 1.327 | 1.332 | 1.322 | 1.321 |
| cold_both | flaml-chemberta | spread | 1.327 | 1.331 | 1.324 | 1.312 |
| cold_both | siamese-ecfp | oracle | 1.403 | 1.121 | 0.932 | 0.783 |
| cold_both | siamese-ecfp | random | 1.403 | 1.401 | 1.392 | 1.400 |
| cold_both | siamese-ecfp | spread | 1.403 | 1.379 | 1.353 | 1.328 |
| cold_both | siamese | oracle | 1.339 | 1.069 | 0.898 | 0.757 |
| cold_both | siamese | random | 1.339 | 1.340 | 1.345 | 1.335 |
| cold_both | siamese | spread | 1.339 | 1.336 | 1.328 | 1.310 |
| cold_both | fusion | oracle | 1.374 | 1.081 | 0.905 | 0.758 |
| cold_both | fusion | random | 1.374 | 1.368 | 1.369 | 1.367 |
| cold_both | fusion | spread | 1.374 | 1.363 | 1.340 | 1.324 |
