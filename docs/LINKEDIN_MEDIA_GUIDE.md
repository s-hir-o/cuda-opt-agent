# LinkedIn Demo Media Guide

This guide shows how to run `cuda-opt-agent` locally and collect clean media for a GitHub/LinkedIn post.

## 1. Prerequisites

For the real CUDA demo path:

- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- Visual Studio Build Tools or Visual Studio C++ toolchain on Windows
- NVIDIA Nsight Compute CLI, `ncu`, if you want profiling media
- Python environment managed by `uv`
- CUDA Samples `Common` headers available at the path configured in `workspace/matrixMul_cli_config.yaml`

The included demo config currently uses:

```yaml
compiler:
  default_arch: "sm_75"
  extra_flags:
    - "-ID:/work/repo/cuda-samples/Common"
```

Change `default_arch` to match your GPU if needed, for example `sm_80`, `sm_86`, or `sm_89`.

## 2. Basic Health Check

Run the test suite:

```powershell
uv run pytest -q
```

Expected result from the current checkout:

```text
660 passed, 5 skipped
```

Good media:

- Screenshot of the passing test line.
- Keep the terminal zoomed in and crop out unrelated folders.

## 3. Show The CLI Surface

Run:

```powershell
uv run python -m cuda_opt_agent.cli.app --help
```

Good media:

- Screenshot showing the available commands: `optimize`, `profile`, `benchmark`, and `history`.
- This proves the project has an actual developer interface, not only internal modules.

## 4. Benchmark The Demo CUDA Harness

From PowerShell on Windows:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && uv run python -m cuda_opt_agent.cli.app benchmark .\workspace\matrixMul_cli_harness.cu --arch sm_75 --warmup 1 --iterations 10 --config .\workspace\matrixMul_cli_config.yaml'
```

Current local sample output:

```text
Benchmarking matrixMul_cli_harness.cu (warmup=1, iterations=10)

Benchmark complete:
  Variant:      matrixMul_cli_harness
  Mean:         0.0901 ms
  Median:       0.0892 ms
  Std Dev:      0.0016 ms
  Min:          0.0890 ms
  Max:          0.0926 ms
  Samples:      10
  P95:          0.0925 ms
```

Good media:

- Screenshot of benchmark statistics.
- Highlight `Median`, `Std Dev`, `Samples`, and `P95`.
- If recording video, start from the command, then show the result table.

## 5. Profile The Demo Harness

Only use this if `ncu` is installed and accessible:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && uv run python -m cuda_opt_agent.cli.app profile .\workspace\matrixMul_cli_harness.cu --arch sm_75 --config .\workspace\matrixMul_cli_config.yaml'
```

Good media:

- Screenshot showing GPU time, SM throughput, DRAM throughput, occupancy, registers/thread, block size, or grid size.
- This is strong media for high-performance and systems-oriented companies.

## 6. Optimization Run

The `optimize` command is the full pipeline, but the included demo config uses `provider: "vllm"`. That means optimization expects an external OpenAI-compatible vLLM server to be running.

Use this only if your vLLM endpoint is available:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && uv run python -m cuda_opt_agent.cli.app optimize .\workspace\matrixMul_cli_harness.cu --entry-point MatrixMulCUDA --arch sm_75 --max-iter 2 --output .\workspace\matrixMul_optimized.cu --config .\workspace\matrixMul_cli_config.yaml'
```

Good media:

- Screenshot of the optimization summary table.
- Screenshot of `workspace/matrixMul_optimized.cu` beside the baseline file.
- If no faster variant is found, present it honestly as evidence of correctness-first evaluation, not as a failed demo.

## 7. Architecture Screenshot

Use a code editor screenshot showing these folders:

```text
cuda_opt_agent/
  agents/
  analysis/
  benchmark/
  build/
  cli/
  correctness/
  cost_models/
  features/
  llm/
  memory/
  profiling/
  search/
  transforms/
```

Good media:

- One screenshot of the package tree.
- One screenshot of `cuda_opt_agent/agents/coordinator.py`.
- One screenshot of `cuda_opt_agent/benchmark/comparator.py`.

## 8. Best LinkedIn Carousel Structure

Use 5 slides:

1. Title: `CUDA Kernel Optimization Workbench`
2. Architecture: profile -> analyze -> transform search -> compile -> correctness -> benchmark -> persist
3. CLI proof: help output or command list
4. Benchmark proof: timing table with median/stddev/P95
5. Engineering discipline: 18.8k+ Python lines, 59 test files, 665 collected tests, 660 passed/5 skipped

## 9. Short Screen Recording Plan

Record a 45-60 second video:

1. Show repository/package tree for 5 seconds.
2. Run `uv run python -m cuda_opt_agent.cli.app --help`.
3. Run the benchmark command.
4. Zoom into the timing result.
5. End on the architecture line:

```text
profile -> analyze -> search transforms -> compile -> validate output -> benchmark -> persist
```

Use OBS Studio, Windows Snipping Tool screen recording, ShareX, or any simple terminal recorder.

## 10. Do Not Show Publicly

Do not show or upload:

- `config/secret_key`
- `config/field_encryption_key`
- `.venv/`
- `.cudenv/`
- `.tmp_pytest*/`
- `workspace/cuda_opt.db`
- generated `.exe` and `.bin` files
- API keys, local tokens, or machine-specific private paths

