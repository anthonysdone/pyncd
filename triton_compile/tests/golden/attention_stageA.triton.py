# Kernel 0
import triton
import triton.language as tl

@triton.jit
def _einops_matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.int32, N: tl.int32, K: tl.int32):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + (offs_m[:, None] * K + (k + offs_k)[None, :])
        b_ptrs = b_ptr + ((k + offs_k)[:, None] * N + offs_n[None, :])
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

# Kernel 1
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols: tl.int32):

    row_idx = tl.program_id(0)
    BLOCK: tl.constexpr = 1024
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    tl.store(y_ptr + row_start + col_offsets, numerator / denominator, mask=mask)

# Kernel 2
import triton
import triton.language as tl

@triton.jit
def _wtril_kernel(x_ptr, y_ptr, n_cols: tl.int32):

    row_idx = tl.program_id(0)
    BLOCK: tl.constexpr = 1024
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    tril_mask = col_offsets <= row_idx
    x_masked = tl.where(tril_mask & mask, x, 0.0)
    denom = tl.sum(x_masked, axis=0) + 1e-8
    tl.store(y_ptr + row_start + col_offsets, x_masked / denom, mask=mask)

# Kernel 3
import triton
import triton.language as tl

@triton.jit
def _einops_matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.int32, N: tl.int32, K: tl.int32):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + (offs_m[:, None] * K + (k + offs_k)[None, :])
        b_ptrs = b_ptr + ((k + offs_k)[:, None] * N + offs_n[None, :])
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

