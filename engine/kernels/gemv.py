"""Fused skinny GEMMs for the decode step (M = batch rows, M <= 64).

One kernel computes y[M, N] = x[M, K] @ W[N, K]^T, reading each weight once
for all rows, with optional fusions that remove whole kernels and their HBM
round trips:

- NORM prologue: x is RMS-normalised while tiles load. The per-row sum of
  squares ``ss`` was accumulated by whichever kernel last wrote the residual
  stream, so no separate norm kernel is needed. Cast placement matches the
  reference: bf16(w * bf16(x * rstd)).
- RESID epilogue: h <- bf16(h + bf16(acc)) (the reference rounds the
  projection, then adds) and atomically accumulates sum(h^2) per row into
  ``ss_out`` for the next norm. Needs the full K per program (no split).
- SWIGLU epilogue: gate and up tiles for the same columns, then
  bf16(bf16(silu(bf16(g))) * bf16(u)).
- F32 epilogue: fp32 accumulator, atomically summed when split over K; the
  consumer rounds to bf16 (the reference's projection rounding) and re-zeroes.
- BF16 epilogue: plain rounded store (LM head logits).

Only the order of fp32 accumulation differs from cuBLAS.
"""

import torch
import triton
import triton.language as tl

EPI_F32, EPI_BF16, EPI_RESID, EPI_SWIGLU = 0, 1, 2, 3


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, y_ptr, nw_ptr, ss_ptr, h_ptr, ss_out_ptr,
    M, N, K, K_PER_SPLIT, eps,
    NORM: tl.constexpr, EPI: tl.constexpr, SPLIT: tl.constexpr, FMA: tl.constexpr,
    MPAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    m = tl.arange(0, MPAD)
    mmask = m < M
    k0 = pid_k * K_PER_SPLIT
    if FMA:
        # M == 1 on CUDA cores: [BN, BK] partial products, one reduction at the
        # end. Many small programs keep enough loads in flight to saturate HBM.
        if NORM:
            rstd0 = tl.math.rsqrt(tl.load(ss_ptr) / K + eps)
        accw = tl.zeros([BN, BK], tl.float32)
        if EPI == 3:
            accw_u = tl.zeros([BN, BK], tl.float32)
        for kk in range(k0, k0 + K_PER_SPLIT, BK):
            k = kk + tl.arange(0, BK)
            xv = tl.load(x_ptr + k).to(tl.float32)
            if NORM:
                nw = tl.load(nw_ptr + k).to(tl.float32)
                xv = (nw * (xv * rstd0).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            w = tl.load(w_ptr + n[:, None] * K + k[None, :]).to(tl.float32)
            accw += w * xv[None, :]
            if EPI == 3:
                wu = tl.load(w_ptr + (n[:, None] + N) * K + k[None, :]).to(tl.float32)
                accw_u += wu * xv[None, :]
        # row 0 of an [MPAD, BN] tile, so the epilogues below serve both paths
        acc = tl.where((m == 0)[:, None], tl.sum(accw, axis=1)[None, :], 0.0)
        if EPI == 3:
            acc_u = tl.where((m == 0)[:, None], tl.sum(accw_u, axis=1)[None, :], 0.0)
    else:
        if NORM:
            ss = tl.load(ss_ptr + m, mask=mmask, other=0.0)
            rstd = tl.math.rsqrt(ss / K + eps)
        acc = tl.zeros([MPAD, BN], tl.float32)
        if EPI == 3:
            acc_u = tl.zeros([MPAD, BN], tl.float32)
        for kk in range(k0, k0 + K_PER_SPLIT, BK):
            k = kk + tl.arange(0, BK)
            x = tl.load(x_ptr + m[:, None] * K + k[None, :], mask=mmask[:, None], other=0.0)
            if NORM:
                nw = tl.load(nw_ptr + k).to(tl.float32)
                xn = (x.to(tl.float32) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
                x = (nw[None, :] * xn).to(tl.bfloat16)
            w = tl.load(w_ptr + n[:, None] * K + k[None, :])
            acc += tl.dot(x, tl.trans(w))
            if EPI == 3:
                wu = tl.load(w_ptr + (n[:, None] + N) * K + k[None, :])
                acc_u += tl.dot(x, tl.trans(wu))

    out = m[:, None] * N + n[None, :]
    omask = mmask[:, None]
    if EPI == 0:
        if SPLIT == 1:
            tl.store(y_ptr + out, acc, mask=omask)
        else:
            tl.atomic_add(y_ptr + out, acc, mask=omask)
    elif EPI == 1:
        tl.store(y_ptr + out, acc.to(tl.bfloat16), mask=omask)
    elif EPI == 2:
        y = acc.to(tl.bfloat16).to(tl.float32)
        h = tl.load(h_ptr + out, mask=omask, other=0.0).to(tl.float32)
        hn = (h + y).to(tl.bfloat16)
        tl.store(h_ptr + out, hn, mask=omask)
        hf = hn.to(tl.float32)
        tl.atomic_add(ss_out_ptr + m, tl.sum(hf * hf, axis=1), mask=mmask)
    else:
        g = acc.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        tl.store(y_ptr + out, (s * u).to(tl.bfloat16), mask=omask)


class Gemv:
    """One fused projection; its launch config is picked by GPU timing at setup.

    x: [M, K] bf16; w: [N, K] (or [2N, K] gate|up for SWIGLU) bf16.
    cfg = (fma, BN, BK, split, num_warps, num_stages).
    """

    def __init__(self, M, N, K, epi, norm, allow_split=False, eps=1e-6):
        self.M, self.N, self.K, self.epi, self.norm, self.eps = M, N, K, epi, norm, eps
        self.mpad = max(16, triton.next_power_of_2(M))
        self.allow_split = allow_split and epi == EPI_F32
        self.cfg = self.candidates()[0]

    def candidates(self):
        """Short list from graph-timed sweeps on H100 (cold weights, CUDA graphs)."""
        out = []
        if self.M == 1:
            for bn, bk, w, st in ((16, 512, 8, 4), (16, 512, 8, 3), (8, 512, 4, 4), (32, 256, 8, 3), (16, 256, 4, 4)):
                if self.epi == EPI_SWIGLU and bn * bk > 4096:
                    bn = bn // 2  # two accumulators
                out.append((True, bn, bk, 1, w, st))
        for bn, bk, split, w, st in ((64, 256, 1, 4, 4), (64, 128, 1, 4, 4), (32, 256, 1, 4, 3),
                                     (32, 128, 1, 4, 4), (16, 128, 1, 4, 4), (64, 256, 2, 4, 4), (32, 128, 2, 4, 4)):
            if split > 1 and not self.allow_split:
                continue
            if self.epi == EPI_SWIGLU and bn == 64 and bk == 256:
                continue
            out.append((False, bn, bk, split, w, st))
        ok = [c for c in dict.fromkeys(out) if self.N % c[1] == 0 and self.K % (c[2] * c[3]) == 0]
        return ok

    def __call__(self, x, w, y=None, nw=None, ss=None, h=None, ss_out=None, cfg=None):
        fma, bn, bk, split, warps, stages = cfg or self.cfg
        dummy = x
        _gemv_kernel[(self.N // bn, split)](
            x, w, y if y is not None else dummy,
            nw if nw is not None else dummy, ss if ss is not None else dummy,
            h if h is not None else dummy, ss_out if ss_out is not None else dummy,
            self.M, self.N, self.K, self.K // split, self.eps,
            NORM=self.norm, EPI=self.epi, SPLIT=split, FMA=fma, MPAD=self.mpad, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages,
        )

    def tune(self, weights, deadline):
        """Pick the fastest candidate by GPU time: one call per real layer weight
        (cold, as in a decode step), captured in a CUDA graph. Eager timing of
        ~10 us kernels measures the host launch path, not the GPU. Outputs go
        to scratch buffers. Stops trying new candidates after ``deadline``
        (time.monotonic()), keeping the best so far."""
        import time
        dev = weights[0].device
        M, N, K = self.M, self.N, self.K
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        y = torch.zeros(M, N, device=dev, dtype=torch.float32 if self.epi == EPI_F32 else torch.bfloat16)
        nw = torch.ones(K, device=dev, dtype=torch.bfloat16)
        ss = torch.full((M,), float(K), device=dev)
        h = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        ss_out = torch.zeros(M, device=dev)
        best = None
        for cfg in self.candidates():
            if time.monotonic() > deadline:
                break
            try:
                self(x, weights[0], y, nw, ss, h, ss_out, cfg=cfg)   # compile outside capture
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for wt in weights:
                        self(x, wt, y, nw, ss, h, ss_out, cfg=cfg)
                g.replay()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                g.replay()
                b.record()
                torch.cuda.synchronize()
                t = a.elapsed_time(b)
                del g
            except Exception:
                torch.cuda.synchronize()
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        if best is not None:
            self.cfg = best[1]
            self.us = best[0] / (2 * len(weights)) * 1e3
        return self.cfg


# ---------------------------------------------------------------------------
# Row kernels around the GEMVs.
# ---------------------------------------------------------------------------
@triton.jit
def _embed_ss_kernel(ids_ptr, emb_ptr, h_ptr, ss_ptr, H, BLOCK: tl.constexpr):
    """h[m] = embed[ids[m]]; ss[m] = sum(h[m]^2) (also zeroes the ss slot)."""
    m = tl.program_id(0)
    tok = tl.load(ids_ptr + m)
    c = tl.arange(0, BLOCK)
    mask = c < H
    row = tl.load(emb_ptr + tok * H + c, mask=mask, other=0.0)
    tl.store(h_ptr + m * H + c, row, mask=mask)
    f = row.to(tl.float32)
    tl.store(ss_ptr + m, tl.sum(f * f, axis=0))


def embed_ss(ids, emb, h, ss):
    M, H = h.shape
    _embed_ss_kernel[(M,)](ids, emb, h, ss, H, BLOCK=triton.next_power_of_2(H), num_warps=8)


@triton.jit
def _qk_norm_rope_acc_kernel(
    acc_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr, zero_ptr, n_zero,
    cache_b_stride, cache_h_stride, eps,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, ZPAD: tl.constexpr,
):
    """Decode (T=1) q/k norm + RoPE from the fp32 QKV accumulator.

    Rounds the projection to bf16 as the reference does, writes q and the
    K/V cache slot, re-zeroes the accumulator slice it consumed, and program
    (0, 0) zeroes the ``zero_ptr`` sum-of-squares buffer for the next norm.
    """
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    ROW: tl.constexpr = (NQ + 2 * NKV) * D
    d = tl.arange(0, D)
    d_rot = (d + HALF) % D
    pos = tl.load(pos_ptr + b).to(tl.int64)
    base = acc_ptr + b * ROW + h * D
    x = tl.load(base + d).to(tl.bfloat16).to(tl.float32)
    xr = tl.load(base + d_rot).to(tl.bfloat16).to(tl.float32)
    tl.store(base + d, tl.zeros([D], tl.float32))
    if h < NQ + NKV:
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
            tl.store(q_out_ptr + b * (NQ * D) + h * D + d, y)
        else:
            tl.store(k_cache_ptr + b * cache_b_stride + (h - NQ) * cache_h_stride + pos * D + d, y)
    else:
        tl.store(v_cache_ptr + b * cache_b_stride + (h - NQ - NKV) * cache_h_stride + pos * D + d,
                 x.to(tl.bfloat16))
    if (b == 0) & (h == 0):
        z = tl.arange(0, ZPAD)
        tl.store(zero_ptr + z, tl.zeros([ZPAD], tl.float32), mask=z < n_zero)


def qk_norm_rope_acc(acc, q_w, k_w, cos, sin, pos, q_out, k_cache, v_cache, zero_buf, eps, nq, nkv, d):
    B = acc.shape[0]
    _qk_norm_rope_acc_kernel[(B, nq + 2 * nkv)](
        acc, q_w, k_w, cos, sin, pos, q_out, k_cache, v_cache, zero_buf, zero_buf.numel(),
        k_cache.stride(0), k_cache.stride(1), eps,
        NQ=nq, NKV=nkv, D=d, ZPAD=max(16, triton.next_power_of_2(zero_buf.numel())), num_warps=1,
    )
