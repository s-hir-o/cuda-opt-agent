# 60 real CUDA CLI tests

Completed 60/60: 5 confirmed improvements, 0 inconclusive, 45 unchanged, 0 confirmed regressions, 10 failures.
Independent CPU-reference checks passed: 300. A successful CLI exit alone is not counted as an optimization.

RTX 2080 Ti, sm_75; twelve kernel families, five sizes each.
Each case runs the real optimize CLI with 18 candidate attempts, then independently recompiles its input and output.
Correctness: all outputs against a NumPy float64 CPU reference at size/seed 0, size+3/seed 1, and size-1/seed 7.
Timing: 12 alternating samples per version, each with 5 warmups and 30 launches timed by CUDA events.
Improved: changed source and bootstrap 95% lower speedup bound >1.01. Regressed: upper bound <0.99.
Unchanged source is never counted as improvement, regardless of measured timing noise.
No LLM credentials were supplied; these runs exercise built-in transforms. Profiling is blocked by GPU-counter permissions.
Confidence intervals describe this local timing sample; kernel families and sizes are related cases, not 60 unrelated algorithms.
Reproduce from the repository root: `.venv\Scripts\python.exe -m workspace.suite60_20261028.run_suite` (resumes saved cases).
Each case directory contains input/output sources, exact CLI arguments, optimizer logs, compiled binaries, output buffers, and result.json.

| Case | Result | Baseline ms | Output ms | Speedup | 95% interval |
|---|---|---:|---:|---:|---|
| matmul_96 | improved | 0.012928 | 0.011099 | 1.165x | 1.032–1.380 |
| matmul_192 | improved | 0.025583 | 0.018913 | 1.353x | 1.322–1.386 |
| matmul_320 | improved | 0.080396 | 0.054346 | 1.479x | 1.477–1.481 |
| matmul_448 | improved | 0.210329 | 0.136911 | 1.536x | 1.534–1.539 |
| matmul_640 | improved | 0.608971 | 0.384301 | 1.585x | 1.538–1.619 |
| matmul_transposed_b_96 | failed | — | — | — | — |
| matmul_transposed_b_192 | unchanged | 0.112894 | 0.113082 | 0.998x | 0.969–1.028 |
| matmul_transposed_b_320 | unchanged | 0.429727 | 0.435062 | 0.988x | 0.976–1.000 |
| matmul_transposed_b_448 | failed | — | — | — | — |
| matmul_transposed_b_640 | unchanged | 2.354513 | 2.339910 | 1.006x | 0.960–1.066 |
| matvec_192 | unchanged | 0.025903 | 0.025538 | 1.014x | 0.985–1.047 |
| matvec_384 | unchanged | 0.046909 | 0.046902 | 1.000x | 0.995–1.005 |
| matvec_640 | unchanged | 0.080718 | 0.080738 | 1.000x | 0.998–1.002 |
| matvec_896 | unchanged | 0.103526 | 0.103724 | 0.998x | 0.994–1.001 |
| matvec_1280 | unchanged | 0.163124 | 0.162966 | 1.001x | 0.999–1.003 |
| row_sum_192 | failed | — | — | — | — |
| row_sum_384 | unchanged | 0.046041 | 0.046038 | 1.000x | 0.996–1.007 |
| row_sum_640 | unchanged | 0.078584 | 0.078737 | 0.998x | 0.994–1.002 |
| row_sum_896 | unchanged | 0.100486 | 0.100561 | 0.999x | 0.996–1.002 |
| row_sum_1280 | failed | — | — | — | — |
| row_dot_192 | unchanged | 0.044277 | 0.044168 | 1.002x | 0.999–1.008 |
| row_dot_384 | unchanged | 0.085603 | 0.085764 | 0.998x | 0.996–1.000 |
| row_dot_640 | unchanged | 0.150173 | 0.150046 | 1.001x | 0.999–1.003 |
| row_dot_896 | unchanged | 0.195852 | 0.196965 | 0.994x | 0.984–1.000 |
| row_dot_1280 | unchanged | 0.305096 | 0.305298 | 0.999x | 0.998–1.000 |
| row_max_192 | unchanged | 0.025264 | 0.025056 | 1.008x | 0.983–1.031 |
| row_max_384 | unchanged | 0.046111 | 0.046044 | 1.001x | 0.997–1.007 |
| row_max_640 | unchanged | 0.079238 | 0.079055 | 1.002x | 0.999–1.007 |
| row_max_896 | unchanged | 0.100781 | 0.100819 | 1.000x | 0.998–1.001 |
| row_max_1280 | unchanged | 0.160171 | 0.160245 | 1.000x | 0.998–1.001 |
| transpose_192 | failed | — | — | — | — |
| transpose_384 | failed | — | — | — | — |
| transpose_640 | unchanged | 0.014875 | 0.015186 | 0.980x | 0.905–1.060 |
| transpose_896 | unchanged | 0.020591 | 0.021162 | 0.973x | 0.948–0.999 |
| transpose_1280 | unchanged | 0.045445 | 0.045497 | 0.999x | 0.995–1.002 |
| stencil9_8192 | unchanged | 0.011150 | 0.011179 | 0.997x | 0.847–1.223 |
| stencil9_32768 | unchanged | 0.010767 | 0.010784 | 0.998x | 0.922–1.088 |
| stencil9_131072 | failed | — | — | — | — |
| stencil9_524288 | failed | — | — | — | — |
| stencil9_2097152 | unchanged | 0.043602 | 0.043500 | 1.002x | 1.000–1.005 |
| conv3x3_192 | unchanged | 0.010508 | 0.010777 | 0.975x | 0.898–1.056 |
| conv3x3_384 | unchanged | 0.012886 | 0.014571 | 0.884x | 0.741–1.009 |
| conv3x3_640 | unchanged | 0.018186 | 0.018026 | 1.009x | 0.967–1.056 |
| conv3x3_896 | unchanged | 0.036282 | 0.036245 | 1.001x | 0.997–1.006 |
| conv3x3_1280 | unchanged | 0.069345 | 0.069269 | 1.001x | 0.999–1.003 |
| saxpy_8192 | unchanged | 0.011070 | 0.011395 | 0.972x | 0.744–1.224 |
| saxpy_32768 | unchanged | 0.011447 | 0.013343 | 0.858x | 0.651–1.122 |
| saxpy_131072 | unchanged | 0.010612 | 0.010509 | 1.010x | 0.894–1.139 |
| saxpy_524288 | unchanged | 0.020448 | 0.016264 | 1.257x | 1.007–1.561 |
| saxpy_2097152 | unchanged | 0.049976 | 0.054102 | 0.924x | 0.819–1.026 |
| poly5_8192 | unchanged | 0.013717 | 0.012208 | 1.124x | 0.786–1.574 |
| poly5_32768 | unchanged | 0.011641 | 0.014964 | 0.778x | 0.563–1.069 |
| poly5_131072 | unchanged | 0.013021 | 0.014492 | 0.898x | 0.643–1.241 |
| poly5_524288 | unchanged | 0.020112 | 0.017398 | 1.156x | 0.887–1.510 |
| poly5_2097152 | unchanged | 0.059843 | 0.061224 | 0.977x | 0.835–1.158 |
| row_softmax_192 | unchanged | 0.114227 | 0.118704 | 0.962x | 0.906–1.023 |
| row_softmax_384 | unchanged | 0.252705 | 0.239872 | 1.053x | 1.014–1.099 |
| row_softmax_640 | failed | — | — | — | — |
| row_softmax_896 | failed | — | — | — | — |
| row_softmax_1280 | unchanged | 0.907619 | 0.911543 | 0.996x | 0.987–1.005 |