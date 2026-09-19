"""Bit-exact fused kernels for the prefill.

The prompt's K/V must equal the ones the judge computes (a teacher-forced
Transformers forward), because a prompt token with sharp attention amplifies
any difference layer over layer. These kernels therefore reproduce the
reference bit for bit, not just its cast placement:

- reductions (the RMSNorm mean of squares) are left to the same torch op the
  reference uses, so their summation order is identical; rsqrt too;
- everything fused here is elementwise multiplies, adds, negation and casts,
  which are exact under IEEE round-to-nearest in any grouping of kernels;
- SiLU uses correctly rounded division and libdevice expf, as torch's CUDA
  kernel does, instead of Triton's fast approximations.
"""

import torch
import triton
import triton.language as tl

try:  # Triton 3.x
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice


@triton.jit
def _add_square_kernel(x_ptr, d_ptr, h_ptr, sq_ptr, n, HAS_DELTA: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0)
    if HAS_DELTA:
        x = (x.to(tl.float32) + tl.load(d_ptr + i, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(h_ptr + i, x, mask=m)
    f = x.to(tl.float32)
    tl.store(sq_ptr + i, f * f, mask=m)


def mean_sq_rstd(x, eps, delta=None):
    """Qwen3RMSNorm's rstd: torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + eps).

    h = x + delta (bf16, the reference's residual add) when delta is given.
    The cast and square are fused (exact); the mean stays torch's own reduction
    so its summation order, and therefore every bit of rstd, is the reference's.
    Returns (h, rstd).
    """
    h = torch.empty_like(x) if delta is not None else x
    sq = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    _add_square_kernel[(triton.cdiv(n, 2048),)](x, delta if delta is not None else x, h, sq, n,
                                                HAS_DELTA=delta is not None, BLOCK=2048, num_warps=8)
    return h, torch.rsqrt(sq.mean(-1, keepdim=True) + eps)


@triton.jit
def _norm_apply_kernel(x_ptr, rstd_ptr, w_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(x_ptr + row * N + c, mask=m, other=0.0).to(tl.float32)
    r = tl.load(rstd_ptr + row)
    w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + row * N + c, (w * xn).to(tl.bfloat16), mask=m)


def norm_apply(x, rstd, weight):
    """weight * (x.float() * rstd).to(bf16), rows of x contiguous; returns a new tensor."""
    N = x.shape[-1]
    rows = x.numel() // N
    y = torch.empty_like(x)
    _norm_apply_kernel[(rows,)](x, rstd, weight, y, N, BLOCK=triton.next_power_of_2(N), num_warps=4)
    return y


@triton.jit
def _qk_rope_kernel(
    src_ptr, rstd_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr, cache_ptr,
    S, H, REP, cache_b_stride, cache_h_stride,
    D: tl.constexpr, NORM: tl.constexpr, ROPE: tl.constexpr, CACHE: tl.constexpr,
):
    """One (token, head) per program. src: [B, S, H, D] (projection output).

    NORM/ROPE: per-head RMSNorm apply + rotary embedding, reference rounding.
    Writes the head to out [B, H*REP, S, D] (REP copies: repeat_kv's layout)
    and, if CACHE, to cache [B, H, cap, D] at position s.
    """
    tok = tl.program_id(0).to(tl.int64)  # b * S + s
    h = tl.program_id(1).to(tl.int64)
    b = tok // S
    s = tok % S
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, D)
    d_rot = (d + HALF) % D
    base = src_ptr + (tok * H + h) * D
    if NORM:
        r = tl.load(rstd_ptr + tok * H + h)
        w = tl.load(w_ptr + d).to(tl.float32)
        wr = tl.load(w_ptr + d_rot).to(tl.float32)
        x = tl.load(base + d).to(tl.float32)
        xr = tl.load(base + d_rot).to(tl.float32)
        xn = (w * (x * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        xrn = (wr * (xr * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        xn = tl.load(base + d).to(tl.float32)
    if ROPE:
        rh = tl.where(d < HALF, -xrn, xrn)
        c = tl.load(cos_ptr + s * D + d).to(tl.float32)
        sn = tl.load(sin_ptr + s * D + d).to(tl.float32)
        a = (xn * c).to(tl.bfloat16).to(tl.float32)
        bb = (rh * sn).to(tl.bfloat16).to(tl.float32)
        y = (a + bb).to(tl.bfloat16)
    else:
        y = xn.to(tl.bfloat16)
    for r_ in range(REP):
        tl.store(out_ptr + ((b * H * REP + h * REP + r_) * S + s) * D + d, y)
    if CACHE:
        tl.store(cache_ptr + b * cache_b_stride + h * cache_h_stride + s * D + d, y)


def qk_rope(src, rstd, weight, cos, sin, rep, cache=None, norm=True, rope=True):
    """src [B, S, H, D] contiguous -> out [B, H*rep, S, D] contiguous (+ cache [B, H, cap, D])."""
    B, S, H, D = src.shape
    out = torch.empty(B, H * rep, S, D, dtype=src.dtype, device=src.device)
    dummy = src
    _qk_rope_kernel[(B * S, H)](
        src, rstd if rstd is not None else dummy, weight if weight is not None else dummy,
        cos if cos is not None else dummy, sin if sin is not None else dummy, out,
        cache if cache is not None else dummy,
        S, H, rep, cache.stride(0) if cache is not None else 0, cache.stride(1) if cache is not None else 0,
        D=D, NORM=norm, ROPE=rope, CACHE=cache is not None, num_warps=1,
    )
    return out


@triton.jit
def _silu_mul_exact_kernel(g_ptr, u_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + i, mask=m, other=0.0).to(tl.float32)
    s = libdevice.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + i, (s * u).to(tl.bfloat16), mask=m)


def silu_mul_exact(g, u):
    """F.silu(g) * u for bf16 tensors, bit-identical to torch."""
    y = torch.empty_like(g)
    n = g.numel()
    _silu_mul_exact_kernel[(triton.cdiv(n, 2048),)](g, u, y, n, BLOCK=2048, num_warps=8)
    return y
