from pathlib import Path
p=Path(__file__).with_name('run_suite.py')
s=p.read_text()
s=s.replace('Thirty real', 'Sixty real').replace('suite30', 'suite60_20261028')
start=s.index('FAMILIES = {'); end=s.index('\nKERNELS =',start)
s=s[:start]+'''FAMILIES = {
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
''' + s[end:]
pos=s.index('HARNESS =')
s=s[:pos]+'''KERNELS.pop("serial_vecadd")
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
''' + s[pos:]
s=s.replace('#include <vector>', '#include <vector>\n#include <cmath>')
s=s.replace('matrix = family in ("matmul", "matmul_transposed_b")', 'matrix = family in MATRIX_OUTPUT and family != "row_softmax"')
s=s.replace('large_input = matrix or family in ("matvec", "row_sum")', 'large_input = family in MATRIX_INPUT')
s=s.replace('"size_t(N)*N" if matrix else "N"', '"size_t(N)*N" if family in MATRIX_OUTPUT else "N"')
s=s.replace('family in ("matmul", "matmul_transposed_b", "matvec", "row_sum")', 'family in MATRIX_INPUT')
s=s.replace('    return a+b', '''    if family == "row_dot": return (a*b).reshape(size,size).sum(axis=1)
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
    raise ValueError(family)''')
s=s.replace('size=size, status="started"', 'size=size, description=DESCRIPTIONS[family], status="started"')
s=s.replace('((size,0),(size+3,1))', '((size,0),(size+3,1),(max(1,size-1),7))')
s=s.replace('actual = np.fromfile(buffer,dtype=np.float32)', 'if buffer.stat().st_size != expected.size*4: raise RuntimeError(f"{label} wrong output byte count")\n                actual = np.fromfile(buffer,dtype=np.float32)')
s=s.replace('30 real CUDA CLI tests','60 real CUDA CLI tests').replace('/30','/60').replace('ALL 30','ALL 60').replace('six kernel families','twelve kernel families').replace('not 30 unrelated','not 60 unrelated')
s=s.replace('at the original size/seed 0 and size+3/seed 1.', 'at size/seed 0, size+3/seed 1, and size-1/seed 7.')
s=s.replace('ROOT = Path(__file__).resolve().parent','ROOT = Path(__file__).resolve().parent')
s=s.replace('results=[]\n    for family', '(ROOT/"manifest.json").write_text(json.dumps([dict(case=f"{family}_{size}",family=family,size=size,description=DESCRIPTIONS[family]) for family,sizes in FAMILIES.items() for size in sizes],indent=2))\n    results=[]\n    for family')
p.write_text(s)
