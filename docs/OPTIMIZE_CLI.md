# CUDA Optimize CLI

Use `cuda-opt optimize` when you have a complete host-runnable CUDA source file
and want the optimizer to write the best validated source variant.

## Command

```powershell
python -m cuda_opt_agent.cli.app optimize .\path\to\kernel.cu `
  --entry-point my_kernel `
  --arch sm_80 `
  --max-iter 20 `
  --provider vllm `
  --output .\workspace\kernel_optimized.cu
```

`--provider` is optional. If omitted, the provider from `config.yaml` is used.

## Input Contract

The input file must be a complete CUDA translation unit, not only a device
kernel. The CLI validates these requirements before starting optimization:

1. It declares the requested kernel entry point:

   ```cuda
   __global__ void my_kernel(...)
   ```

2. It includes a host executable harness:

   ```cuda
   int main(int argc, char **argv) {
       ...
   }
   ```

3. The executable supports the correctness contract:

   ```text
   executable --output <path>
   ```

   Both the baseline and each candidate executable must write the same raw
   numeric output buffer format to that path. The default comparator reads
   `float32` values and applies the configured absolute and relative tolerances.

## Output Behavior

When `--output` is provided, the CLI writes the best available variant source to
that path.

- If a faster correct variant is found, that source is written.
- If no faster correct variant is found but the baseline ran, the baseline
  source is written and the CLI reports that no faster variant was found.
- If no iterations are produced or the output path cannot be written, the
  command exits with code `1`.

Generated candidates are required to preserve the original entry point, host
`main(...)` harness, and `--output <path>` correctness behavior.
