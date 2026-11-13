# conformal + selective prediction (alpha = 0.1)

## Split conformal — calibrated on val, applied to test (90% target)

| split | arm | seeds | half-width | coverage | test RMSE |
|---|---|---|---|---|---|
| cold_drug | flaml-ecfp | 3 | 1.520 ± 0.037 | 0.902 ± 0.004 | 0.923 |
| cold_drug | flaml-chemberta | 3 | 1.682 ± 0.036 | 0.903 ± 0.001 | 1.020 |
| cold_drug | siamese-ecfp | 3 | 1.739 ± 0.027 | 0.904 ± 0.004 | 1.047 |
| cold_drug | siamese | 3 | 1.700 ± 0.015 | 0.900 ± 0.003 | 1.043 |
| cold_drug | fusion | 3 | 1.667 ± 0.017 | 0.897 ± 0.003 | 1.026 |
| scaffold | flaml-ecfp | 3 | 1.582 ± 0.033 | 0.903 ± 0.001 | 0.959 |
| scaffold | flaml-chemberta | 3 | 1.752 ± 0.033 | 0.904 ± 0.002 | 1.052 |
| scaffold | siamese-ecfp | 3 | 1.835 ± 0.002 | 0.903 ± 0.002 | 1.110 |
| scaffold | siamese | 3 | 1.818 ± 0.014 | 0.904 ± 0.002 | 1.093 |
| scaffold | fusion | 3 | 1.779 ± 0.007 | 0.905 ± 0.002 | 1.071 |
| cold_target | flaml-ecfp | 3 | 2.199 ± 0.037 | 0.900 ± 0.017 | 1.374 |
| cold_target | flaml-chemberta | 3 | 2.164 ± 0.070 | 0.899 ± 0.017 | 1.335 |
| cold_target | siamese-ecfp | 3 | 2.349 ± 0.061 | 0.918 ± 0.012 | 1.376 |
| cold_target | siamese | 3 | 2.186 ± 0.066 | 0.886 ± 0.021 | 1.384 |
| cold_target | fusion | 3 | 2.219 ± 0.104 | 0.902 ± 0.043 | 1.334 |
| cold_both | flaml-ecfp | 3 | 2.149 ± 0.005 | 0.891 ± 0.003 | 1.393 |
| cold_both | flaml-chemberta | 3 | 2.193 ± 0.044 | 0.900 ± 0.013 | 1.372 |
| cold_both | siamese-ecfp | 3 | 2.424 ± 0.047 | 0.916 ± 0.015 | 1.416 |
| cold_both | siamese | 3 | 2.250 ± 0.036 | 0.902 ± 0.008 | 1.390 |
| cold_both | fusion | 3 | 2.318 ± 0.044 | 0.908 ± 0.006 | 1.414 |

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
| cold_drug | 25% | 26% | 28% |
| scaffold | 22% | 22% | 23% |
| cold_target | 11% | 10% | 9% |
| cold_both | 6% | 10% | 7% |
| cold_drug | flaml-ecfp | oracle | 0.923 | 0.663 | 0.535 | 0.438 |
| cold_drug | flaml-ecfp | random | 0.923 | 0.919 | 0.915 | 0.916 |
| cold_drug | flaml-ecfp | spread | 0.923 | 0.878 | 0.853 | 0.825 |
| cold_drug | flaml-chemberta | oracle | 1.020 | 0.765 | 0.631 | 0.525 |
| cold_drug | flaml-chemberta | random | 1.020 | 1.017 | 1.017 | 1.019 |
| cold_drug | flaml-chemberta | spread | 1.020 | 0.954 | 0.916 | 0.879 |
| cold_drug | siamese-ecfp | oracle | 1.047 | 0.761 | 0.617 | 0.508 |
| cold_drug | siamese-ecfp | random | 1.047 | 1.052 | 1.051 | 1.050 |
| cold_drug | siamese-ecfp | spread | 1.047 | 0.963 | 0.922 | 0.880 |
| cold_drug | siamese | oracle | 1.043 | 0.755 | 0.613 | 0.504 |
| cold_drug | siamese | random | 1.043 | 1.041 | 1.043 | 1.045 |
| cold_drug | siamese | spread | 1.043 | 0.961 | 0.918 | 0.877 |
| cold_drug | fusion | oracle | 1.026 | 0.733 | 0.591 | 0.484 |
| cold_drug | fusion | random | 1.026 | 1.026 | 1.028 | 1.029 |
| cold_drug | fusion | spread | 1.026 | 0.949 | 0.908 | 0.869 |
| scaffold | flaml-ecfp | oracle | 0.959 | 0.706 | 0.575 | 0.472 |
| scaffold | flaml-ecfp | random | 0.959 | 0.959 | 0.961 | 0.962 |
| scaffold | flaml-ecfp | spread | 0.959 | 0.923 | 0.905 | 0.883 |
| scaffold | flaml-chemberta | oracle | 1.052 | 0.799 | 0.659 | 0.547 |
| scaffold | flaml-chemberta | random | 1.052 | 1.054 | 1.054 | 1.053 |
| scaffold | flaml-chemberta | spread | 1.052 | 1.002 | 0.972 | 0.945 |
| scaffold | siamese-ecfp | oracle | 1.110 | 0.824 | 0.675 | 0.558 |
| scaffold | siamese-ecfp | random | 1.110 | 1.110 | 1.108 | 1.109 |
| scaffold | siamese-ecfp | spread | 1.110 | 1.031 | 0.993 | 0.956 |
| scaffold | siamese | oracle | 1.093 | 0.802 | 0.653 | 0.538 |
| scaffold | siamese | random | 1.093 | 1.090 | 1.088 | 1.087 |
| scaffold | siamese | spread | 1.093 | 1.019 | 0.981 | 0.947 |
| scaffold | fusion | oracle | 1.071 | 0.782 | 0.636 | 0.522 |
| scaffold | fusion | random | 1.071 | 1.072 | 1.072 | 1.072 |
| scaffold | fusion | spread | 1.071 | 1.002 | 0.970 | 0.938 |
| cold_target | flaml-ecfp | oracle | 1.374 | 1.124 | 0.964 | 0.823 |
| cold_target | flaml-ecfp | random | 1.374 | 1.376 | 1.375 | 1.373 |
| cold_target | flaml-ecfp | spread | 1.374 | 1.349 | 1.333 | 1.320 |
| cold_target | flaml-chemberta | oracle | 1.335 | 1.089 | 0.930 | 0.792 |
| cold_target | flaml-chemberta | random | 1.335 | 1.333 | 1.334 | 1.332 |
| cold_target | flaml-chemberta | spread | 1.335 | 1.303 | 1.293 | 1.287 |
| cold_target | siamese-ecfp | oracle | 1.376 | 1.098 | 0.929 | 0.787 |
| cold_target | siamese-ecfp | random | 1.376 | 1.378 | 1.377 | 1.377 |
| cold_target | siamese-ecfp | spread | 1.376 | 1.347 | 1.331 | 1.320 |
| cold_target | siamese | oracle | 1.384 | 1.104 | 0.930 | 0.784 |
| cold_target | siamese | random | 1.384 | 1.384 | 1.385 | 1.386 |
| cold_target | siamese | spread | 1.384 | 1.346 | 1.328 | 1.317 |
| cold_target | fusion | oracle | 1.334 | 1.085 | 0.920 | 0.777 |
| cold_target | fusion | random | 1.334 | 1.335 | 1.335 | 1.334 |
| cold_target | fusion | spread | 1.334 | 1.313 | 1.301 | 1.293 |
| cold_both | flaml-ecfp | oracle | 1.393 | 1.124 | 0.964 | 0.825 |
| cold_both | flaml-ecfp | random | 1.393 | 1.397 | 1.403 | 1.393 |
| cold_both | flaml-ecfp | spread | 1.393 | 1.373 | 1.352 | 1.355 |
| cold_both | flaml-chemberta | oracle | 1.372 | 1.092 | 0.928 | 0.790 |
| cold_both | flaml-chemberta | random | 1.372 | 1.375 | 1.381 | 1.383 |
| cold_both | flaml-chemberta | spread | 1.372 | 1.355 | 1.331 | 1.336 |
| cold_both | siamese-ecfp | oracle | 1.416 | 1.110 | 0.928 | 0.786 |
| cold_both | siamese-ecfp | random | 1.416 | 1.421 | 1.422 | 1.422 |
| cold_both | siamese-ecfp | spread | 1.416 | 1.393 | 1.362 | 1.353 |
| cold_both | siamese | oracle | 1.390 | 1.087 | 0.915 | 0.770 |
| cold_both | siamese | random | 1.390 | 1.394 | 1.384 | 1.379 |
| cold_both | siamese | spread | 1.390 | 1.380 | 1.356 | 1.361 |
| cold_both | fusion | oracle | 1.414 | 1.112 | 0.941 | 0.797 |
| cold_both | fusion | random | 1.414 | 1.396 | 1.390 | 1.386 |
| cold_both | fusion | spread | 1.414 | 1.397 | 1.362 | 1.358 |
