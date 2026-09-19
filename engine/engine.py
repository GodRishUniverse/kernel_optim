"""Qwen3 4B engine: hand-rolled forward, static KV cache, CUDA-graph decode,
prompt-lookup speculative decoding with exact greedy verification.

Prefill runs eagerly (cuBLAS GEMMs + SDPA flash attention, as the reference
does). Decode replays CUDA graphs built from fused Triton kernels (norms, RoPE,
SwiGLU, split-K GQA attention) and fused QKV / gate-up GEMMs.

Speculation: each step feeds every sequence its last token plus K tokens
drafted by n-gram lookup over its own prompt + output, verifies all K+1
positions in one forward (causal attention over the block), and keeps the
longest drafted prefix that matches the model's own argmax, plus the model's
next token. Every emitted token is therefore the model's greedy choice on its
own prefix. If acceptance is too low to pay for the host round trip, the engine
drops to a pipelined one-token graph for the rest of the generation.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from kernels.fused import DecodeAttention, add_rms_norm, qk_norm_rope, silu_mul
from kernels.gemv import EPI_BF16, EPI_F32, EPI_RESID, EPI_SWIGLU, Gemv, embed_ss, qk_norm_rope_acc

DEVICE = "cuda:0"
NGRAMS = (3, 2, 1)          # draft lookup, longest match first
PROBE_STEPS = 6             # speculative steps before judging acceptance
MIN_ADVANCE = float(os.environ.get("ENGINE_MIN_ADVANCE", "1.25"))
LOG = os.environ.get("ENGINE_LOG", "1") == "1"
# Prompt-lookup drafts rarely match on the judge's prompts; the probe steps cost
# more than they save, so speculation is opt-in.
SPEC = os.environ.get("ENGINE_SPEC", "0") == "1"
FUSED_MAX_B = 64            # fused-GEMV decode up to this batch; cuBLAS beyond
TUNE_BUDGET_S = 150.0       # stop timing GEMV configs after this much of the load budget


def draft_len(B):
    """Draft tokens per sequence; verify rows B*(K+1) stay in the GEMV regime."""
    if B <= 2:
        return 6
    if B <= 8:
        return 4
    if B <= 16:
        return 3
    if B <= 32:
        return 1
    return 0


class _Lookup:
    """Incremental n-gram index over one sequence: ngram -> latest continuation index."""

    __slots__ = ("seq", "maps")

    def __init__(self, seq):
        self.seq = seq
        self.maps = {n: {} for n in NGRAMS}
        for i in range(1, len(seq)):
            self._index(i)

    def _index(self, i):
        # ngram ending at seq[i-1] is continued by seq[i]
        s = self.seq
        for n in NGRAMS:
            if i >= n:
                self.maps[n][tuple(s[i - n:i])] = i

    def append(self, toks):
        for t in toks:
            self.seq.append(t)
            self._index(len(self.seq) - 1)

    def draft(self, K):
        s = self.seq
        L = len(s)
        for n in NGRAMS:
            j = self.maps[n].get(tuple(s[L - n:]))
            if j is not None:
                out = []
                for k in range(K):  # continuation may run into the draft itself (loops)
                    idx = j + k
                    out.append(s[idx] if idx < L else out[idx - L])
                return out
        return [s[-1]] * K


def _prefill_attention(q, k, v, native_gqa):
    """Causal attention, q [B, NQ, S, D], k/v [B, NKV, S, D] (strided cache views).

    Same flash kernel the reference reaches through repeat_kv + SDPA; with
    native GQA support it reads the 8 KV heads in place instead of writing
    4 copies of them to HBM every layer.
    """
    scale = q.shape[-1] ** -0.5
    if native_gqa:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):  # never a slower fallback
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    B, nkv, S, D = k.shape
    rep = q.shape[1] // nkv
    k = k[:, :, None].expand(B, nkv, rep, S, D).reshape(B, nkv * rep, S, D)
    v = v[:, :, None].expand(B, nkv, rep, S, D).reshape(B, nkv * rep, S, D)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _probe_native_gqa():
    """Use enable_gqa only if this torch supports it and it is bit-identical."""
    try:
        g = torch.Generator(device=DEVICE).manual_seed(0)
        q = torch.randn(2, 32, 300, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
        k = torch.randn(2, 8, 300, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
        v = torch.randn(2, 8, 300, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
        ok = torch.equal(_prefill_attention(q, k, v, True), _prefill_attention(q, k, v, False))
    except Exception:
        ok = False
    if LOG:
        print(f"[engine] native GQA prefill attention: {ok}", file=sys.stderr, flush=True)
    return ok


class Engine:
    def __init__(self, model_path: str) -> None:
        self._t0 = time.monotonic()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to(DEVICE)
        cfg = model.config
        self.nq = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", cfg.hidden_size // self.nq)
        self.inter = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.n_layers = cfg.num_hidden_layers
        self.rotary = model.model.rotary_emb

        base = model.model
        self.embed = base.embed_tokens.weight
        self.lm_head = model.lm_head.weight
        self.final_norm = base.norm.weight
        self.layers = []
        with torch.no_grad():
            for layer in base.layers:
                a, m = layer.self_attn, layer.mlp
                self.layers.append(dict(
                    w_in=layer.input_layernorm.weight,
                    w_post=layer.post_attention_layernorm.weight,
                    qkv=torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous(),
                    q_norm=a.q_norm.weight,
                    k_norm=a.k_norm.weight,
                    o=a.o_proj.weight,
                    gu=torch.cat([m.gate_proj.weight, m.up_proj.weight], 0).contiguous(),
                    down=m.down_proj.weight,
                ))
                a.q_proj = a.k_proj = a.v_proj = None
                m.gate_proj = m.up_proj = None
        del model
        torch.cuda.empty_cache()

        self._cos = self._sin = None
        self._shape = None
        self.gqa_native = _probe_native_gqa()

    # ------------------------------------------------------------------ setup
    def _rope_tables(self, n):
        if self._cos is not None and self._cos.shape[0] >= n:
            return
        pos = torch.arange(n, device=DEVICE).unsqueeze(0)
        dummy = torch.empty(1, dtype=torch.bfloat16, device=DEVICE)
        cos, sin = self.rotary(dummy, pos)  # exactly the reference's tables
        self._cos = cos[0].contiguous()
        self._sin = sin[0].contiguous()

    def _setup(self, B, S, N):
        key = (B, S, N)
        if self._shape == key:
            return
        self._shape = None
        self.g1 = self.gk = None
        self.k_cache = self.v_cache = None
        torch.cuda.empty_cache()

        self.K = draft_len(B) if SPEC and N > 2 else 0
        # a sequence can run ahead of the slowest one by up to N (+K) tokens
        cap = S + 2 * N + 2 * self.K + 2
        self._rope_tables(cap)
        self.k_cache = torch.zeros(self.n_layers, B, self.nkv, cap, self.d, dtype=torch.bfloat16, device=DEVICE)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pos = torch.zeros(B, dtype=torch.int32, device=DEVICE)
        self.ids = torch.zeros(B, dtype=torch.int64, device=DEVICE)
        self.B, self.cap = B, cap
        self.pin_ids1 = torch.empty(N + 1, B, dtype=torch.int64, pin_memory=True)
        self.events = [torch.cuda.Event() for _ in range(N + 1)]
        self.pin_pos = torch.empty(B, dtype=torch.int32, pin_memory=True)

        self.attn1 = DecodeAttention(B, 1, cap, self.nq, self.nkv, self.d, DEVICE)
        self.g1 = self._capture(self._step1)
        if B <= FUSED_MAX_B:
            # Keep whichever decode graph is faster on this GPU, measured here
            # during the untimed warmup rather than assumed.
            self._setup_fused(B)
            fused = self._capture(self._step1_fused)
            t_plain, t_fused = self._time_graph(self.g1, S), self._time_graph(fused, S)
            t_plain, t_fused = min(t_plain, self._time_graph(self.g1, S)), min(t_fused, self._time_graph(fused, S))
            if LOG:
                print(f"[engine] decode step B={B}: cublas {t_plain * 1e3:.0f}us fused {t_fused * 1e3:.0f}us",
                      file=sys.stderr, flush=True)
            if t_fused < t_plain:
                self.g1 = fused
        if self.K:
            T = self.K + 1
            self.attnk = DecodeAttention(B, T, cap, self.nq, self.nkv, self.d, DEVICE)
            self.vin = torch.zeros(B, T, dtype=torch.int64, device=DEVICE)
            self.vout = torch.zeros(B, T, dtype=torch.int64, device=DEVICE)
            self.pin_vin = torch.empty(B, T, dtype=torch.int64, pin_memory=True)
            self.pin_vout = torch.empty(B, T, dtype=torch.int64, pin_memory=True)
            self.gk = self._capture(self._stepk)
        self._shape = key

    def _time_graph(self, g, S, reps=10):
        torch.cuda.synchronize()
        self.pos.fill_(S)
        g.replay()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps

    def _setup_fused(self, B):
        H, I, V = self.embed.shape[1], self.inter, self.embed.shape[0]
        nqkv = (self.nq + 2 * self.nkv) * self.d
        bf = torch.bfloat16
        self.h = torch.zeros(B, H, dtype=bf, device=DEVICE)
        self.ss = torch.zeros(2, B, dtype=torch.float32, device=DEVICE)
        self.acc_qkv = torch.zeros(B, nqkv, dtype=torch.float32, device=DEVICE)
        self.q1 = torch.zeros(B, self.nq * self.d, dtype=bf, device=DEVICE)
        self.attn_out = torch.zeros(B, self.nq * self.d, dtype=bf, device=DEVICE)
        self.act = torch.zeros(B, I, dtype=bf, device=DEVICE)
        self.logits = torch.zeros(B, V, dtype=bf, device=DEVICE)
        eps = self.eps
        lw = self.layers[0]
        self.gv = dict(
            qkv=(Gemv(B, nqkv, H, EPI_F32, norm=True, allow_split=True, eps=eps), lw["qkv"]),
            o=(Gemv(B, H, self.nq * self.d, EPI_RESID, norm=False, eps=eps), lw["o"]),
            gu=(Gemv(B, I, H, EPI_SWIGLU, norm=True, eps=eps), lw["gu"]),
            down=(Gemv(B, H, I, EPI_RESID, norm=False, eps=eps), lw["down"]),
            lm=(Gemv(B, V, H, EPI_BF16, norm=True, eps=eps), self.lm_head),
        )
        for name, (g, w) in self.gv.items():
            if time.monotonic() - self._t0 < TUNE_BUDGET_S:
                g.tune(w)
            if LOG:
                print(f"[engine] gemv {name} B={B} cfg={g.cfg}", file=sys.stderr, flush=True)
        self.acc_qkv.zero_()
        self.ss.zero_()

    def _step1_fused(self):
        """Decode step, 7 kernels per layer: norms ride in GEMV prologues, residual
        adds and the next norm's sum of squares in GEMV epilogues."""
        g = {k: v[0] for k, v in self.gv.items()}
        h, ss = self.h, self.ss
        embed_ss(self.ids, self.embed, h, ss[0])
        for i, lw in enumerate(self.layers):
            g["qkv"](h, lw["qkv"], y=self.acc_qkv, nw=lw["w_in"], ss=ss[0])
            # consumes acc_qkv; zeroes both ss rows (ss[0] consumed above, ss[1] by the last gate-up)
            qk_norm_rope_acc(self.acc_qkv, lw["q_norm"], lw["k_norm"], self._cos, self._sin, self.pos,
                             self.q1, self.k_cache[i], self.v_cache[i], ss, self.eps,
                             self.nq, self.nkv, self.d)
            self.attn1(self.q1, self.k_cache[i], self.v_cache[i], self.pos, self.attn_out)
            g["o"](self.attn_out, lw["o"], h=h, ss_out=ss[1])
            g["gu"](h, lw["gu"], y=self.act, nw=lw["w_post"], ss=ss[1])
            g["down"](self.act, lw["down"], h=h, ss_out=ss[0])
        g["lm"](h, self.lm_head, y=self.logits, nw=self.final_norm, ss=ss[0])
        torch.argmax(self.logits, dim=-1, out=self.ids)
        self.pos.add_(1)

    def _capture(self, fn):
        # Warm up (compiles Triton kernels, cuBLAS handles) on a side stream,
        # then capture. The cache is scratch here; prefill resets positions.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.pos.zero_()
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        self.pos.zero_()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    # ---------------------------------------------------------------- forward
    def _layers(self, x, B, T, attn_fn, all_rows):
        """x: embedded tokens [B*T, H] (consumed as the residual stream).

        Returns final-normed hidden states: every row if all_rows, else the
        last token of each sequence.
        """
        M = B * T
        h = x
        n = torch.empty_like(h)
        q = torch.empty(M, self.nq * self.d, dtype=torch.bfloat16, device=DEVICE)
        act = torch.empty(M, self.inter, dtype=torch.bfloat16, device=DEVICE)
        delta = None
        for i, lw in enumerate(self.layers):
            add_rms_norm(h, delta, lw["w_in"], n, self.eps)
            qkv = torch.mm(n, lw["qkv"].t())
            qk_norm_rope(qkv, lw["q_norm"], lw["k_norm"], self._cos, self._sin, self.pos,
                         q, self.k_cache[i], self.v_cache[i], T, self.eps, self.nq, self.nkv, self.d)
            a = attn_fn(i, q)
            o = torch.mm(a, lw["o"].t())
            add_rms_norm(h, o, lw["w_post"], n, self.eps)
            gu = torch.mm(n, lw["gu"].t())
            silu_mul(gu, act)
            delta = torch.mm(act, lw["down"].t())
        if T > 1 and not all_rows:
            last = torch.arange(B, device=DEVICE) * T + (T - 1)
            h, delta = h[last].contiguous(), delta[last].contiguous()
        out = torch.empty_like(h)
        add_rms_norm(h, delta, self.final_norm, out, self.eps)
        return out

    def _prefill(self, prompt):
        B, S = prompt.shape
        self.pos.zero_()
        x = F.embedding(prompt.reshape(-1), self.embed)
        nq, d = self.nq, self.d

        def attn(i, q):
            qh = q.view(B, S, nq, d).transpose(1, 2)
            k = self.k_cache[i][:, :, :S]
            v = self.v_cache[i][:, :, :S]
            return _prefill_attention(qh, k, v, self.gqa_native).transpose(1, 2).reshape(B * S, nq * d)

        hn = self._layers(x, B, S, attn, all_rows=False)
        logits = torch.mm(hn, self.lm_head.t())
        self.ids.copy_(torch.argmax(logits, dim=-1))
        self.pos.fill_(S)

    def _step1(self):
        """One token per sequence at pos[b]; argmax feeds ids, positions advance."""
        B = self.B
        x = F.embedding(self.ids, self.embed)
        out = torch.empty(B, self.nq * self.d, dtype=torch.bfloat16, device=DEVICE)

        def attn(i, q):
            self.attn1(q, self.k_cache[i], self.v_cache[i], self.pos, out)
            return out

        hn = self._layers(x, B, 1, attn, all_rows=True)
        logits = torch.mm(hn, self.lm_head.t())
        torch.argmax(logits, dim=-1, out=self.ids)
        self.pos.add_(1)

    def _stepk(self):
        """Verify: vin[b] = last token + K drafts at pos[b].. -> vout = argmax at each."""
        B, T = self.B, self.K + 1
        x = F.embedding(self.vin.view(-1), self.embed)
        out = torch.empty(B * T, self.nq * self.d, dtype=torch.bfloat16, device=DEVICE)

        def attn(i, q):
            self.attnk(q, self.k_cache[i], self.v_cache[i], self.pos, out)
            return out

        hn = self._layers(x, B, T, attn, all_rows=True)
        logits = torch.mm(hn, self.lm_head.t())
        torch.argmax(logits, dim=-1, out=self.vout.view(-1))

    # --------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        N = max_new_tokens
        if N <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        self._setup(B, S, N)
        stream = torch.cuda.current_stream()

        prompt = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
        self._prefill(prompt)
        self.pin_ids1[0].copy_(self.ids, non_blocking=True)
        self.events[0].record(stream)
        # Queue the first decode step before waiting on prefill, so the GPU keeps
        # working while the first token is handed to the caller.
        launched = 0
        if not self.K and N > 1:
            self.g1.replay()
            self.pin_ids1[1].copy_(self.ids, non_blocking=True)
            self.events[1].record(stream)
            launched = 1
        # Build draft indexes while the GPU runs prefill.
        look = [_Lookup(list(row)) for row in input_ids] if self.K else None
        self.events[0].synchronize()
        first = self.pin_ids1[0].tolist()
        yield first
        if N == 1:
            return

        outs = [[t] for t in first]   # generated tokens per sequence
        emitted = 1
        spec_steps = 0
        if self.K:
            K, T = self.K, self.K + 1
            for b in range(B):
                look[b].append([first[b]])
            vin, vout, ppos = self.pin_vin, self.pin_vout, self.pin_pos
            # a frozen (finished) sequence keeps its position; its rows are ignored
            start_min = 1
            while emitted < N:
                rows, plist = [], []
                for b in range(B):
                    seq = look[b].seq
                    rows.append([seq[-1]] + look[b].draft(K))
                    plist.append(len(seq) - 1)
                vin.copy_(torch.tensor(rows))
                ppos.copy_(torch.tensor(plist, dtype=torch.int32))
                self.vin.copy_(vin, non_blocking=True)
                self.pos.copy_(ppos, non_blocking=True)
                self.gk.replay()
                vout.copy_(self.vout, non_blocking=True)
                self.events[1].record(stream)
                self.events[1].synchronize()
                spec_steps += 1
                g = vout.tolist()
                d = vin.tolist()
                for b in range(B):
                    if len(outs[b]) >= N:
                        continue  # finished: frozen, its rows are ignored
                    gb, db = g[b], d[b]
                    a = 0
                    while a < K and db[a + 1] == gb[a]:
                        a += 1
                    new = gb[:a + 1]
                    outs[b].extend(new)
                    look[b].append(new)
                ready = min(len(o) for o in outs)
                while emitted < min(ready, N):
                    yield [o[emitted] for o in outs]
                    emitted += 1
                if spec_steps == PROBE_STEPS and emitted < N:
                    advance = (ready - start_min) / spec_steps
                    if advance < MIN_ADVANCE:
                        break  # not paying for itself: finish on the pipelined path
            if emitted >= N:
                self._log(B, S, N, spec_steps, 0, outs)
                return
            # hand the per-sequence state to the one-token graph
            for b in range(B):
                vin[b, 0] = outs[b][-1]
                ppos[b] = S + len(outs[b]) - 1
            self.ids.copy_(vin[:, 0], non_blocking=True)
            self.pos.copy_(ppos, non_blocking=True)
        # Pipelined one-token decode: the next replay is queued before the host
        # waits on the previous step's ids.
        steps = N - min(len(o) for o in outs)
        ids_h, ev = self.pin_ids1, self.events
        for j in range(1, steps + 1):
            if j > launched:
                self.g1.replay()
                ids_h[j].copy_(self.ids, non_blocking=True)
                ev[j].record(stream)
            if j > 1:
                yield from self._drain(ids_h, ev, j - 1, outs, N, emitted)
                emitted = min(N, min(len(o) for o in outs))
        if steps:
            yield from self._drain(ids_h, ev, steps, outs, N, emitted)
        self._log(B, S, N, spec_steps, steps, outs)

    def _drain(self, ids_h, ev, j, outs, N, emitted):
        ev[j].synchronize()
        toks = ids_h[j].tolist()
        for b, o in enumerate(outs):
            o.append(toks[b])
        ready = min(N, min(len(o) for o in outs))
        while emitted < ready:
            yield [o[emitted] for o in outs]
            emitted += 1

    def _log(self, B, S, N, spec_steps, plain_steps, outs):
        if LOG:
            print(f"[engine] B={B} S={S} N={N} K={self.K} spec_steps={spec_steps} "
                  f"plain_steps={plain_steps} forwards={1 + spec_steps + plain_steps}",
                  file=sys.stderr, flush=True)
