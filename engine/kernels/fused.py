"""Fused Triton kernels for the Qwen3 forward.

Every kernel mirrors the cast placement of Transformers 4.51.3: arithmetic runs
in fp32 and is rounded to bf16 exactly where the reference rounds, so fusing
only reorders reductions and never reformulates the math.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# RMSNorm, optionally preceded by a residual add (residual updated in place).
# ---------------------------------------------------------------------------
@triton.jit
def _add_rms_norm_kernel(
    res_ptr, delta_ptr, w_ptr, out_ptr, n_cols, eps,
    HAS_DELTA: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row * n_cols + cols
    x = tl.load(res_ptr + offs, mask=mask, other=0.0)
    if HAS_DELTA:
        # reference: hidden = residual + hidden, rounded to bf16
        d = tl.load(delta_ptr + offs, mask=mask, other=0.0)
        x = (x.to(tl.float32) + d.to(tl.float32)).to(tl.bfloat16)
        tl.store(res_ptr + offs, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    normed = (xf * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (w.to(tl.float32) * normed.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + offs, y, mask=mask)


def add_rms_norm(res, delta, weight, out, eps):
    """out = rmsnorm(res + delta); res <- res + delta. Rows contiguous [M, N]."""
    n_rows, n_cols = res.shape
    _add_rms_norm_kernel[(n_rows,)](
        res, delta if delta is not None else res, weight, out, n_cols, eps,
        HAS_DELTA=delta is not None, BLOCK=triton.next_power_of_2(n_cols),
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Per-head q/k RMSNorm + RoPE, writing K/V straight into the cache.
# qkv: [M, (NQ + 2*NKV) * D] with M = B*T rows; token row m is batch m // T,
# position pos[b] + m % T, with pos[B] read from device memory.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    T, cache_b_stride, cache_h_stride, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    ROW: tl.constexpr = (NQ + 2 * NKV) * D
    d = tl.arange(0, D)
    d_rot = (d + HALF) % D
    b = m // T
    pos = tl.load(pos_ptr + b).to(tl.int64) + m % T
    base = qkv_ptr + m * ROW + h * D
    if h < NQ + NKV:
        x = tl.load(base + d).to(tl.float32)
        xr = tl.load(base + d_rot).to(tl.float32)
        if h < NQ:
            w = tl.load(qw_ptr + d).to(tl.float32)
            wr = tl.load(qw_ptr + d_rot).to(tl.float32)
        else:
            w = tl.load(kw_ptr + d).to(tl.float32)
            wr = tl.load(kw_ptr + d_rot).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        xn = (w * (x * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        xrn = (wr * (xr * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        rh = tl.where(d < HALF, -xrn, xrn)
        c = tl.load(cos_ptr + pos * D + d).to(tl.float32)
        s = tl.load(sin_ptr + pos * D + d).to(tl.float32)
        a = (xn * c).to(tl.bfloat16).to(tl.float32)
        bb = (rh * s).to(tl.bfloat16).to(tl.float32)
        y = (a + bb).to(tl.bfloat16)
        if h < NQ:
            tl.store(q_out_ptr + m * (NQ * D) + h * D + d, y)
        else:
            kh = h - NQ
            tl.store(k_cache_ptr + b * cache_b_stride + kh * cache_h_stride + pos * D + d, y)
    else:
        vh = h - NQ - NKV
        v = tl.load(base + d)
        tl.store(v_cache_ptr + b * cache_b_stride + vh * cache_h_stride + pos * D + d, v)


def qk_norm_rope(qkv, q_w, k_w, cos, sin, pos, q_out, k_cache, v_cache, T, eps, nq, nkv, d):
    """k_cache/v_cache: one layer, [B, NKV, cap, D] contiguous."""
    M = qkv.shape[0]
    _qk_norm_rope_kernel[(M, nq + 2 * nkv)](
        qkv, q_w, k_w, cos, sin, pos, q_out, k_cache, v_cache,
        T, k_cache.stride(0), k_cache.stride(1), eps,
        NQ=nq, NKV=nkv, D=d, num_warps=1,
    )


# ---------------------------------------------------------------------------
# SwiGLU: out = bf16(bf16(silu(gate)) * up), gate/up interleaved as [M, 2*I].
# ---------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, I, BLOCK: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(gu_ptr + m * 2 * I + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + m * 2 * I + I + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + m * I + cols, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul(gu, out):
    M, two_i = gu.shape
    I = two_i // 2
    BLOCK = 1024
    _silu_mul_kernel[(M, triton.cdiv(I, BLOCK))](gu, out, I, BLOCK=BLOCK, num_warps=4)


# ---------------------------------------------------------------------------
# Decode / verify attention: T query tokens per sequence at positions
# pos[b] .. pos[b]+T-1, causal among themselves, against a fixed-capacity
# cache. GQA-aware split-K flash-decoding: one program reads a KV head's slice
# once for all T * GROUP query rows that share it. Positions live in device
# memory so the step can be replayed from a CUDA graph.
# q/out rows are laid out [(b*T + t), NQ*D].
# ---------------------------------------------------------------------------
@triton.jit
def _decode_attn_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, o_ptr, ml_ptr,
    cache_b_stride, cache_h_stride, chunk, scale,
    T: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr,
    GROUP: tl.constexpr, RPAD: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = (bh // NKV).to(tl.int64)
    kh = bh % NKV
    p0 = tl.load(pos_ptr + b).to(tl.int32)
    start = split * chunk
    end = tl.minimum(start + chunk, p0 + T)

    r = tl.arange(0, RPAD)
    t = r // GROUP
    qh = kh * GROUP + r % GROUP
    rmask = r < T * GROUP
    row = (b * T + t) * NQ + qh
    d = tl.arange(0, D)
    q = tl.load(q_ptr + row[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
    limit = p0 + t  # last visible key per query row

    k_base = k_ptr + b * cache_b_stride + kh * cache_h_stride
    v_base = v_ptr + b * cache_b_stride + kh * cache_h_stride
    m_i = tl.full([RPAD], -1.0e30, tl.float32)
    l_i = tl.zeros([RPAD], tl.float32)
    acc = tl.zeros([RPAD, D], tl.float32)
    n = tl.arange(0, BLOCK_N)
    for s0 in range(start, end, BLOCK_N):
        idx = s0 + n
        nmask = idx < end
        k = tl.load(k_base + idx[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(v_base + idx[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(idx[None, :] <= limit[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    part = row * NSPLIT + split
    tl.store(o_ptr + part[:, None] * D + d[None, :], acc, mask=rmask[:, None])
    tl.store(ml_ptr + part * 2, m_i, mask=rmask)
    tl.store(ml_ptr + part * 2 + 1, l_i, mask=rmask)


@triton.jit
def _decode_combine_kernel(o_ptr, ml_ptr, out_ptr, D: tl.constexpr, NSPLIT: tl.constexpr, SPAD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)  # (b*T + t) * NQ + head
    sp = tl.arange(0, SPAD)
    smask = sp < NSPLIT
    d = tl.arange(0, D)
    m = tl.load(ml_ptr + (row * NSPLIT + sp) * 2, mask=smask, other=-1.0e30)
    l = tl.load(ml_ptr + (row * NSPLIT + sp) * 2 + 1, mask=smask, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.exp(m - M)
    L = tl.sum(w * l, axis=0)
    o = tl.load(o_ptr + (row * NSPLIT + sp)[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
    res = tl.sum(o * w[:, None], axis=0) / L
    tl.store(out_ptr + row * D + d, res.to(tl.bfloat16))


class DecodeAttention:
    """Preallocated split-K attention for T query tokens per sequence, fixed (B, T, cap)."""

    BLOCK_N = 64

    def __init__(self, B, T, cap, nq, nkv, d, device, target_ctas=264):
        self.B, self.T, self.nq, self.nkv, self.d = B, T, nq, nkv, d
        max_split = max(1, triton.cdiv(cap, 128))
        nsplit = max(1, min(max_split, triton.cdiv(target_ctas, B * nkv)))
        chunk = triton.cdiv(triton.cdiv(cap, nsplit), self.BLOCK_N) * self.BLOCK_N
        self.nsplit = triton.cdiv(cap, chunk)
        self.chunk = chunk
        rows = B * T * nq
        self.o_part = torch.empty(rows * self.nsplit * d, dtype=torch.float32, device=device)
        self.ml = torch.empty(rows * self.nsplit * 2, dtype=torch.float32, device=device)
        self.scale = d ** -0.5

    def __call__(self, q, k_cache, v_cache, pos, out):
        """q: [B*T, NQ*D]; k/v_cache: [B, NKV, cap, D]; pos: [B] int32; out: [B*T, NQ*D]."""
        group = self.nq // self.nkv
        _decode_attn_kernel[(self.B * self.nkv, self.nsplit)](
            q, k_cache, v_cache, pos, self.o_part, self.ml,
            k_cache.stride(0), k_cache.stride(1), self.chunk, self.scale,
            T=self.T, NQ=self.nq, NKV=self.nkv, D=self.d, NSPLIT=self.nsplit,
            GROUP=group, RPAD=max(16, triton.next_power_of_2(self.T * group)),
            BLOCK_N=self.BLOCK_N, num_warps=4, num_stages=2,
        )
        _decode_combine_kernel[(self.B * self.T * self.nq,)](
            self.o_part, self.ml, out, D=self.d, NSPLIT=self.nsplit,
            SPAD=max(2, triton.next_power_of_2(self.nsplit)), num_warps=1,
        )
