"""Sixty real CLI optimization runs, with independent CPU and timing checks.
Run from repository root: .venv/Scripts/python.exe -m workspace.suite60_20261028.run_suite
Results are saved after every case; rerunning resumes completed cases.
"""
import asyncio
import csv
import json
import hashlib
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

from cuda_opt_agent.build.windows_toolchain import ensure_windows_msvc_environment
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.core.config import CompilerConfig
from cuda_opt_agent.schemas.kernel import KernelSource, KernelLanguage

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
FAMILIES = {
    "matmul": [96, 192, 320, 448, 640],
    "matmul_transposed_b": [96, 192, 320, 448, 640],
    "matvec": [192, 384, 640, 896, 1280],
    "row_sum": [192, 384, 640, 896, 1280],
    "row_dot": [192, 384, 640, 896, 1280],
    "row_max": [192, 384, 640, 896, 1280],
    "transpose": [192, 384, 640, 896, 1280],
    "stencil9": [8192, 32768, 131072, 524288, 2097152],
    "conv3x3": [192, 384, 640, 896, 1280],
    "saxpy": [8192, 32768, 131072, 524288, 2097152],
    "poly5": [8192, 32768, 131072, 524288, 2097152],
    "row_softmax": [192, 384, 640, 896, 1280],
}
MATRIX_OUTPUT = {"matmul", "matmul_transposed_b", "transpose", "conv3x3", "row_softmax"}
MATRIX_INPUT = MATRIX_OUTPUT | {"matvec", "row_sum", "row_dot", "row_max"}
DESCRIPTIONS = {
 "matmul": "Direct one-thread-per-output GEMM; no shared-memory tiling.",
 "matmul_transposed_b": "Pairwise row dot products; each thread walks two rows.",
 "matvec": "One thread per matrix row; serial dot accumulation.",
 "row_sum": "One thread per row; serial reduction with strided inter-thread reads.",
 "row_dot": "Batched row dot products; serial reduction per thread.",
 "row_max": "One thread per row; serial maximum reduction.",
 "transpose": "Direct global-memory transpose without a shared-memory tile.",
 "stencil9": "Periodic nine-point 1D stencil with direct loads and modulo indexing.",
 "conv3x3": "Zero-padded 3x3 box convolution with direct neighborhood loads.",
 "saxpy": "Scalar pointwise SAXPY; no explicit vectorized loads.",
 "poly5": "Pointwise degree-five polynomial using a short Horner loop.",
 "row_softmax": "Numerically stable softmax; serial max/sum/output passes per row.",
}

KERNELS = {
    "matmul": """
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
""",
    "matmul_transposed_b": """
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * N + col] = sum;
    }
""",
    "matvec": """
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k];
        }
        C[i] = sum;
    }
""",
    "row_sum": """
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k];
        }
        C[i] = sum;
    }
""",
    "stencil9": """
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int k = -4; k <= 4; k++) {
            sum += A[(i + k + N) % N];
        }
        C[i] = sum;
    }
""",
    "serial_vecadd": """
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }
""",
}
KERNELS.pop("serial_vecadd")
KERNELS.update({
 "row_dot": """int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) { float sum = 0.0f; for (int k = 0; k < N; k++) {
 sum += A[i*N+k] * B[i*N+k]; } C[i]=sum; }""",
 "row_max": """int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) { float value = -INFINITY; for (int k = 0; k < N; k++) {
 value = fmaxf(value,A[i*N+k]); } C[i]=value; }""",
 "transpose": """int row = blockIdx.y * blockDim.y + threadIdx.y;
 int col = blockIdx.x * blockDim.x + threadIdx.x;
 if (row < N && col < N) C[col*N+row] = A[row*N+col];""",
 "conv3x3": """int row = blockIdx.y * blockDim.y + threadIdx.y;
 int col = blockIdx.x * blockDim.x + threadIdx.x;
 if (row < N && col < N) { float sum=0.0f;
 for(int dy=-1;dy<=1;dy++) { for(int dx=-1;dx<=1;dx++) {
 int y=row+dy, x=col+dx;
 if(y>=0 && y<N && x>=0 && x<N) sum += A[y*N+x]/9.0f;
 }} C[row*N+col]=sum; }""",
 "saxpy": """int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) { C[i] = 1.75f * A[i] + B[i]; }""",
 "poly5": """int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) { float value=0.125f; for(int k=0;k<5;k++) {
 value = value * A[i] + B[i]; } C[i]=value; }""",
 "row_softmax": """int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < N) { float maximum=-INFINITY;
 for(int k=0;k<N;k++) maximum=fmaxf(maximum,A[i*N+k]);
 float total=0.0f; for(int k=0;k<N;k++) total+=expf(A[i*N+k]-maximum);
 for(int k=0;k<N;k++) C[i*N+k]=expf(A[i*N+k]-maximum)/total;
 }""",
})
HARNESS = r'''
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(2); } } while(0)
uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
__global__ void work(float *C, const float *A, const float *B, int N) {
@KERNEL@
}
int main(int argc,char **argv) {
    int N=@SIZE@, seed=0;
    const char *output=nullptr;
    for(int arg=1;arg<argc;++arg) {
        if(!strcmp(argv[arg],"--output") && arg+1<argc) output=argv[++arg];
        else if(!strcmp(argv[arg],"--size") && arg+1<argc) N=atoi(argv[++arg]);
        else if(!strcmp(argv[arg],"--seed") && arg+1<argc) seed=atoi(argv[++arg]);
    }
    size_t inputs=@INPUTS@, outputs=@OUTPUTS@;
    std::vector<float> A(inputs),B(inputs),C(outputs);
    for(size_t j=0;j<inputs;++j) {
        A[j]=float(int(mix(uint32_t(j)+uint32_t(seed)*1000003u+17u)%2048u)-1024)/1024.0f;
        B[j]=float(int(mix(uint32_t(j)+uint32_t(seed)*2000003u+991u)%2048u)-1024)/1024.0f;
    }
    float *a,*b,*c;
    CHECK(cudaMalloc(&a,inputs*sizeof(float))); CHECK(cudaMalloc(&b,inputs*sizeof(float)));
    CHECK(cudaMalloc(&c,outputs*sizeof(float)));
    CHECK(cudaMemcpy(a,A.data(),inputs*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b,B.data(),inputs*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemset(c,255,outputs*sizeof(float)));
    @LAUNCH@
    for(int w=0;w<5;++w) work<<<grid,block>>>(c,a,b,N);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for(int repeat=0;repeat<30;++repeat) work<<<grid,block>>>(c,a,b,N);
    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop)); CHECK(cudaGetLastError());
    float elapsed; CHECK(cudaEventElapsedTime(&elapsed,start,stop));
    CHECK(cudaMemcpy(C.data(),c,outputs*sizeof(float),cudaMemcpyDeviceToHost));
    if(output) {
        FILE *f=fopen(output,"wb"); if(!f) return 3;
        size_t written=fwrite(C.data(),sizeof(float),outputs,f); fclose(f);
        if(written!=outputs) return 4;
    }
    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    printf("%.9f\n",elapsed/30.0f);
    return 0;
}
'''


def source_for(family, size):
    matrix = family in MATRIX_OUTPUT and family != "row_softmax"
    large_input = family in MATRIX_INPUT
    launch = "dim3 block(16,16), grid((N+15)/16,(N+15)/16);" if matrix else (
        "dim3 block(1), grid(1);" if family == "serial_vecadd" else
        "dim3 block(128), grid((N+127)/128);")
    return HARNESS.replace("@KERNEL@", KERNELS[family]).replace("@SIZE@", str(size)).replace(
        "@INPUTS@", "size_t(N)*N" if large_input else "N").replace(
        "@OUTPUTS@", "size_t(N)*N" if family in MATRIX_OUTPUT else "N").replace("@LAUNCH@", launch)


def reference(family, size, seed):
    count = size*size if family in MATRIX_INPUT else size
    j = np.arange(count, dtype=np.uint32)
    def mix(x):
        x = x ^ (x >> 16); x = x * np.uint32(0x7feb352d)
        x = x ^ (x >> 15); x = x * np.uint32(0x846ca68b)
        return x ^ (x >> 16)
    a = ((mix(j+np.uint32(seed*1000003+17))%2048).astype(np.float64)-1024)/1024
    b = ((mix(j+np.uint32(seed*2000003+991))%2048).astype(np.float64)-1024)/1024
    if family == "matmul": return (a.reshape(size,size) @ b.reshape(size,size)).ravel()
    if family == "matmul_transposed_b": return (a.reshape(size,size) @ b.reshape(size,size).T).ravel()
    if family == "matvec": return a.reshape(size,size) @ b[:size]
    if family == "row_sum": return a.reshape(size,size).sum(axis=1)
    if family == "stencil9": return sum(np.roll(a, k) for k in range(-4,5))
    if family == "row_dot": return (a*b).reshape(size,size).sum(axis=1)
    if family == "row_max": return a.reshape(size,size).max(axis=1)
    if family == "transpose": return a.reshape(size,size).T.ravel()
    if family == "conv3x3":
        padded=np.pad(a.reshape(size,size),1)
        return sum(padded[y:y+size,x:x+size]/9 for y in range(3) for x in range(3)).ravel()
    if family == "saxpy": return 1.75*a+b
    if family == "poly5":
        value=np.full(size,0.125)
        for _ in range(5): value=value*a+b
        return value
    if family == "row_softmax":
        values=a.reshape(size,size)
        ex=np.exp(values-values.max(axis=1,keepdims=True))
        return (ex/ex.sum(axis=1,keepdims=True)).ravel()
    raise ValueError(family)


def execute(binary, *args):
    p = subprocess.run([binary, *map(str,args)], capture_output=True, text=True, timeout=120)
    if p.returncode: raise RuntimeError(f"Executable exit {p.returncode}: {p.stderr[-1500:]}")
    value = float(p.stdout.strip().splitlines()[-1])
    if not np.isfinite(value) or value <= 0: raise RuntimeError("Invalid timing sample")
    return value


def host_contract(source):
    """Everything except the work kernel body must remain byte-for-byte identical."""
    start=source.index("__global__ void work")
    opening=source.index("{",start)
    depth=1; end=opening+1
    while depth:
        if source[end] == "{": depth+=1
        elif source[end] == "}": depth-=1
        end+=1
    return source[:opening]+source[end:]


async def run_case(family, size, reuse_cli=False):
    name = f"{family}_{size}"
    folder = ROOT/name
    folder.mkdir(exist_ok=True)
    result = dict(case=name, family=family, size=size, description=DESCRIPTIONS[family], status="started")
    started = time.monotonic()
    try:
        original = source_for(family,size)
        (folder/"input.cu").write_text(original)
        config = dict(compiler=dict(default_arch="sm_75"),
                      benchmark=dict(warmup_iterations=2,timing_iterations=10,use_cuda_events=True,timeout_seconds=120),
                      optimization=dict(convergence_patience=25,min_speedup_threshold=1.01),
                      llm=dict(provider="claude"),
                      database=dict(url=f"sqlite:///{(folder/'runs.db').as_posix()}"))
        # JSON is a YAML subset, so the CLI can load this directly.
        (folder/"config.yaml").write_text(json.dumps(config,indent=2))
        command = [sys.executable,"-m","cuda_opt_agent.cli.app","optimize",str(folder/"input.cu"),
                   "--entry-point","work","--arch","sm_75","--max-iter","18",
                   "--config",str(folder/"config.yaml"),"--output",str(folder/"output.cu")]
        (folder/"command.json").write_text(json.dumps(command,indent=2))
        if not reuse_cli:
            with (folder/"optimize.log").open("w",encoding="utf-8") as log:
                proc = subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        else:
            previous=json.loads((folder/"result.json").read_text())
            if not (folder/"initial_guard_result.json").exists():
                (folder/"initial_guard_result.json").write_text(json.dumps(previous,indent=2))
            proc=subprocess.CompletedProcess(command,previous["cli_exit_code"])
            result["reused_completed_cli"] = True
        result["cli_exit_code"] = proc.returncode
        logtext = (folder/"optimize.log").read_text(encoding="utf-8",errors="replace")
        result["profiling_blocked"] = "ERR_NVGPUCTRPERM" in logtext
        best = re.findall(r"coordinator.done.*?best_speedup=([0-9.]+).*?best_variant=(\S+)",logtext)
        if best: result.update(cli_speedup=float(best[-1][0]),selected_variant=best[-1][1])
        if proc.returncode or not (folder/"output.cu").is_file():
            raise RuntimeError("CLI failed or produced no output; see optimize.log")
        output = (folder/"output.cu").read_text()
        result["host_contract_unchanged"] = host_contract(output) == host_contract(original)
        def strip_unroll(text):
            return re.sub(r"(?m)^[ \t]*#pragma[ \t]+unroll[ \t]+[1-9][0-9]*[ \t]*\n", "", text)
        result["host_semantics_unchanged"] = strip_unroll(host_contract(output)) == strip_unroll(host_contract(original))
        result["host_pragmas_changed"] = not result["host_contract_unchanged"] and result["host_semantics_unchanged"]
        result["measured_source"] = "exact_cli_output"
        if not result["host_semantics_unchanged"]: raise RuntimeError("Candidate changed host harness beyond numeric unroll pragmas")
        result["input_sha256"] = hashlib.sha256(original.encode()).hexdigest()
        result["output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
        result["source_changed"] = output != original
        runner = NVCCRunner(CompilerConfig(default_arch="sm_75"))
        binaries = {}
        for label, code in (("baseline",original),("optimized",output)):
            kernel = KernelSource(name=label+".cu", language=KernelLanguage.CUDA,source_code=code,entry_point="work")
            compiled = await runner.compile_executable(kernel,output_dir=str(folder/label))
            if not compiled.success: raise RuntimeError(f"{label} compile failed: {compiled.stderr[-2000:]}")
            binaries[label] = compiled.binary_path
        checks = []
        for check_size, seed in ((size,0),(size+3,1),(max(1,size-1),7)):
            expected = reference(family,check_size,seed)
            for label,binary in binaries.items():
                buffer = folder/f"{label}_{check_size}.bin"
                execute(binary,"--size",check_size,"--seed",seed,"--output",buffer)
                if buffer.stat().st_size != expected.size*4: raise RuntimeError(f"{label} wrong output byte count")
                actual = np.fromfile(buffer,dtype=np.float32)
                if actual.shape != expected.shape: raise RuntimeError(f"{label} wrong output length")
                if not np.isfinite(actual).all(): raise RuntimeError(f"{label} nonfinite output")
                np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
                checks.append(dict(version=label,size=check_size,seed=seed,elements=len(actual),
                                   max_error=float(np.max(np.abs(actual-expected)))))
        result["cpu_checks"] = checks
        samples = {label:[] for label in binaries}
        for i in range(12):
            order = list(binaries) if i%2==0 else list(reversed(binaries))
            for label in order: samples[label].append(execute(binaries[label]))
        stats = {label:dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),
                            stddev_ms=statistics.stdev(v),samples_ms=v) for label,v in samples.items()}
        result["timings"] = stats
        ratio = stats["baseline"]["mean_ms"]/stats["optimized"]["mean_ms"]
        result["speedup"] = ratio
        # Bootstrap independent samples: a confidence interval for ratio of means.
        rng = np.random.default_rng(42)
        x = np.array(samples["baseline"]); y = np.array(samples["optimized"])
        ratios = x[rng.integers(0,12,(10000,12))].mean(axis=1)/y[rng.integers(0,12,(10000,12))].mean(axis=1)
        low,high = np.quantile(ratios,[.025,.975])
        result["speedup_ci95"] = [float(low),float(high)]
        result["status"] = ("unchanged" if not result["source_changed"] else
                            "improved" if low>1.01 else "regressed" if high<.99 else "inconclusive")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)[-3000:]
    result["elapsed_seconds"] = time.monotonic()-started
    (folder/"result.json").write_text(json.dumps(result,indent=2))
    return result


def save_summary(results):
    (ROOT/"results.json").write_text(json.dumps(results,indent=2))
    fields = ["case","status","cli_exit_code","source_changed","cli_speedup","speedup","baseline_ms","optimized_ms","ci_low","ci_high","selected_variant"]
    with (ROOT/"results.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in results:
            row = {key:r.get(key,"") for key in fields}
            if "timings" in r:
                row.update(baseline_ms=r["timings"]["baseline"]["mean_ms"],optimized_ms=r["timings"]["optimized"]["mean_ms"],ci_low=r["speedup_ci95"][0],ci_high=r["speedup_ci95"][1])
            writer.writerow(row)
    counts = {status:sum(r["status"] == status for r in results) for status in
              ("improved","inconclusive","unchanged","regressed","failed")}
    checks = sum(len(r.get("cpu_checks",[])) for r in results)
    lines = ["# 60 real CUDA CLI tests", "",
             f"Completed {len(results)}/60: {counts['improved']} confirmed improvements, "
             f"{counts['inconclusive']} inconclusive, {counts['unchanged']} unchanged, "
             f"{counts['regressed']} confirmed regressions, {counts['failed']} failures.",
             f"Independent CPU-reference checks passed: {checks}. A successful CLI exit alone is not counted as an optimization.", "",
             "RTX 2080 Ti, sm_75; twelve kernel families, five sizes each.",
             "Each case runs the real optimize CLI with 18 candidate attempts, then independently recompiles its input and output.",
             "Correctness: all outputs against a NumPy float64 CPU reference at size/seed 0, size+3/seed 1, and size-1/seed 7.",
             "Timing: 12 alternating samples per version, each with 5 warmups and 30 launches timed by CUDA events.",
             "Improved: changed source and bootstrap 95% lower speedup bound >1.01. Regressed: upper bound <0.99.",
             "Unchanged source is never counted as improvement, regardless of measured timing noise.",
             "No LLM credentials were supplied; these runs exercise built-in transforms. Profiling is blocked by GPU-counter permissions.",
             "Confidence intervals describe this local timing sample; kernel families and sizes are related cases, not 60 unrelated algorithms.",
             "Reproduce from the repository root: `.venv\\Scripts\\python.exe -m workspace.suite60_20261028.run_suite` (resumes saved cases).",
             "Each case directory contains input/output sources, exact CLI arguments, optimizer logs, compiled binaries, output buffers, and result.json.", "",
             "| Case | Result | Baseline ms | Output ms | Speedup | 95% interval |", "|---|---|---:|---:|---:|---|"]
    for r in results:
        t=r.get("timings")
        lines.append(f"| {r['case']} | {r['status']} | " + (f"{t['baseline']['mean_ms']:.6f} | {t['optimized']['mean_ms']:.6f} | {r['speedup']:.3f}x | {r['speedup_ci95'][0]:.3f}–{r['speedup_ci95'][1]:.3f} |" if t else "— | — | — | — |"))
    (ROOT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


async def main():
    assert ensure_windows_msvc_environment(), "MSVC unavailable"
    (ROOT/"manifest.json").write_text(json.dumps([dict(case=f"{family}_{size}",family=family,size=size,description=DESCRIPTIONS[family]) for family,sizes in FAMILIES.items() for size in sizes],indent=2))
    results=[]
    for family,sizes in FAMILIES.items():
        for size in sizes:
            existing=ROOT/f"{family}_{size}"/"result.json"
            if existing.exists() and "codec can't decode" not in existing.read_text(): r=json.loads(existing.read_text())
            else:
                print(f"START {len(results)+1}/60 {family}_{size}",flush=True)
                r=await run_case(family,size)
            results.append(r); save_summary(results)
            print(f"DONE {len(results)}/60 {r['case']}: {r['status']} speedup={r.get('speedup')} error={r.get('error','')[:180]}",flush=True)
    print("ALL 60 COMPLETE",flush=True)


if __name__ == "__main__":
    asyncio.run(main())
