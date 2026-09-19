"""Qwen3 4B engine: hand-rolled forward, static KV cache, CUDA-graph decode.

Prefill runs eagerly (cuBLAS GEMMs + SDPA flash attention, as the reference
does). Each decode step is one CUDA graph replay: fused Triton kernels for
norms / RoPE / SwiGLU / split-K GQA attention, fused QKV and gate-up GEMMs,
and an in-graph argmax that feeds the next step. Token ids are copied to host
one step behind the GPU so the host never stalls the device.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from kernels.fused import DecodeAttention, add_rms_norm, qk_norm_rope, silu_mul

DEVICE = "cuda:0"


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to(DEVICE)
        cfg = model.config
        self.nq = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", cfg.hidden_size // self.nq)
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.n_layers = cfg.num_hidden_layers
        self.max_pos = cfg.max_position_embeddings
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
                # drop the now-duplicated projection weights
                a.q_proj = a.k_proj = a.v_proj = None
                m.gate_proj = m.up_proj = None
        del model
        torch.cuda.empty_cache()

        self._cos = self._sin = None
        self._shape = None
        self.graph = None
        self._host = None

    # ------------------------------------------------------------------ setup
    def _rope_tables(self, n):
        if self._cos is not None and self._cos.shape[0] >= n:
            return
        pos = torch.arange(n, device=DEVICE).unsqueeze(0)
        dummy = torch.empty(1, dtype=torch.bfloat16, device=DEVICE)
        cos, sin = self.rotary(dummy, pos)  # exactly the reference's tables
        self._cos = cos[0].contiguous()
        self._sin = sin[0].contiguous()

    def _setup(self, B, S, max_new):
        cap = S + max_new
        key = (B, cap)
        if self._shape == key:
            return
        self.graph = None
        self._shape = None
        self.k_cache = self.v_cache = None
        torch.cuda.empty_cache()
        self._rope_tables(cap)
        L, nkv, d = self.n_layers, self.nkv, self.d
        self.k_cache = torch.zeros(L, B, nkv, cap, d, dtype=torch.bfloat16, device=DEVICE)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pos = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        self.ids = torch.zeros(B, dtype=torch.int64, device=DEVICE)
        self.attn = DecodeAttention(B, cap, self.nq, nkv, d, DEVICE)
        self.B, self.cap = B, cap
        self._capture()
        self._shape = key

    # ---------------------------------------------------------------- forward
    def _layers(self, x, B, T, attn_fn):
        """x: embedded tokens [B*T, H]; returns final-normed hidden of last token per sequence."""
        M = B * T
        h = x  # residual stream, updated in place
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
        if T > 1:
            last = torch.arange(B, device=DEVICE) * T + (T - 1)
            h, delta = h[last].contiguous(), delta[last].contiguous()
        out = torch.empty_like(h)
        add_rms_norm(h, delta, self.final_norm, out, self.eps)
        return out

    def _prefill(self, prompt):
        B, S = prompt.shape
        self.pos.zero_()
        x = F.embedding(prompt.reshape(-1), self.embed)
        nq, nkv, d = self.nq, self.nkv, self.d
        rep = nq // nkv

        def attn(i, q):
            qh = q.view(B, S, nq, d).transpose(1, 2)
            k = self.k_cache[i][:, :, :S]
            v = self.v_cache[i][:, :, :S]
            k = k[:, :, None].expand(B, nkv, rep, S, d).reshape(B, nq, S, d)
            v = v[:, :, None].expand(B, nkv, rep, S, d).reshape(B, nq, S, d)
            o = F.scaled_dot_product_attention(qh, k, v, is_causal=True, scale=d ** -0.5)
            return o.transpose(1, 2).reshape(B * S, nq * d)

        hn = self._layers(x, B, S, attn)
        logits = torch.mm(hn, self.lm_head.t())
        self.ids.copy_(torch.argmax(logits, dim=-1))
        self.pos.fill_(S)

    def _decode_step(self):
        B = self.B
        x = F.embedding(self.ids, self.embed)
        out = torch.empty(B, self.nq * self.d, dtype=torch.bfloat16, device=DEVICE)

        def attn(i, q):
            self.attn(q, self.k_cache[i], self.v_cache[i], self.pos, out)
            return out

        hn = self._layers(x, B, 1, attn)
        logits = torch.mm(hn, self.lm_head.t())
        torch.argmax(logits, dim=-1, out=self.ids)
        self.pos.add_(1)

    def _capture(self):
        # Warm up (compiles Triton kernels, cuBLAS handles) on a side stream,
        # then capture. The cache is scratch here; it is reset by prefill.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.pos.zero_()
                self._decode_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.pos.zero_()
        with torch.cuda.graph(self.graph):
            self._decode_step()
        torch.cuda.synchronize()

    # --------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        self._setup(B, S, max_new_tokens)
        prompt = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
        if self._host is None or self._host.shape[0] < max_new_tokens or self._host.shape[1] != B:
            self._host = torch.empty(max_new_tokens, B, dtype=torch.int64, pin_memory=True)
            self._events = [torch.cuda.Event() for _ in range(max_new_tokens)]
        host, events = self._host, self._events

        stream = torch.cuda.current_stream()
        self._prefill(prompt)
        for step in range(max_new_tokens):
            # tokens for `step` are in self.ids; stash them, then launch the next step
            host[step].copy_(self.ids, non_blocking=True)  # stream-ordered before the replay overwrites ids
            events[step].record(stream)
            if step + 1 < max_new_tokens:
                self.graph.replay()
            events[step].synchronize()
            yield host[step].tolist()
