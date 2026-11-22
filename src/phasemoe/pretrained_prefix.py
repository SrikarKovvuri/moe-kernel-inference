"""Exact first-layer router prefix using the optional Transformers OLMoE modules.

Later layers and expert MLP weights cannot affect the first router invocation.
This prefix is useful for prefill occupancy only, not generation or later layers.
"""

import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from torch import nn


class FirstLayerRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeAttention,
            OlmoeRMSNorm,
            OlmoeRotaryEmbedding,
        )

        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.self_attn = OlmoeAttention(config, layer_idx=0)
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)

    def to_inference(self, *, device="cpu", dtype=torch.bfloat16):
        """Match from_pretrained: BF16 parameters, freshly computed FP32 RoPE buffer."""
        self.to(device=device, dtype=dtype)
        # Casting an already-rounded BF16 frequency buffer back to FP32 is lossy.
        self.rotary_emb = type(self.rotary_emb)(config=self.config).to(device=device)
        return self.eval()

    @torch.inference_mode()
    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("prefix supports one unpadded, nonempty prompt at a time")
        hidden = self.embed_tokens(input_ids)
        length = input_ids.shape[1]
        positions = torch.arange(length, device=hidden.device).unsqueeze(0)
        causal = torch.full(
            (length, length),
            torch.finfo(hidden.dtype).min,
            device=hidden.device,
            dtype=hidden.dtype,
        ).triu(1)[None, None]
        attended, _, _ = self.self_attn(
            self.input_layernorm(hidden),
            attention_mask=causal,
            position_ids=positions,
            position_embeddings=self.rotary_emb(hidden, positions),
            use_cache=False,
        )
        normalized = self.post_attention_layernorm(hidden + attended)
        return self.gate(normalized.reshape(-1, self.config.hidden_size))

    @staticmethod
    def checkpoint_key(name):
        if name.startswith("embed_tokens."):
            return "model." + name
        if name.startswith("gate."):
            return "model.layers.0.mlp." + name
        return "model.layers.0." + name


def read_range(url, start, end):
    """Read an exact inclusive range; never fall back to downloading a whole shard."""
    import requests

    # The query separates cache entries for different ranges on redirecting proxies.
    url += ("&" if "?" in url else "?") + f"phasemoe_range={start}-{end}"
    with requests.get(
        url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 180), stream=True
    ) as response:
        response.raise_for_status()
        expected = f"bytes {start}-{end}/"
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
            expected
        ):
            raise RuntimeError("server did not honor the exact byte range; download stopped")
        payload = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            payload.extend(chunk)
            if len(payload) > end - start + 1:
                raise RuntimeError("range response exceeded the requested size")
        if len(payload) != end - start + 1:
            raise RuntimeError("truncated range response")
        return payload


def fetch_prefix_weights(model_id, revision, names, cache_dir):
    """Explicit network operation for a pinned public checkpoint, cached as safetensors."""
    from huggingface_hub import hf_hub_download, hf_hub_url
    from safetensors.torch import load_file, save_file

    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("use a resolved 40-character checkpoint commit")
    cache = Path(cache_dir) / model_id.replace("/", "--") / revision
    weights_path, manifest_path = cache / "first-router.safetensors", cache / "manifest.json"
    if weights_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != manifest["cache_sha256"]:
            raise ValueError("cached prefix weights failed their checksum")
        return load_file(str(weights_path)), manifest
    index = json.loads(
        Path(
            hf_hub_download(model_id, "model.safetensors.index.json", revision=revision)
        ).read_text()
    )
    headers, tensors, records = {}, {}, {}
    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    for name in names:
        shard = index["weight_map"][name]
        url = hf_hub_url(model_id, shard, revision=revision)
        if shard not in headers:
            size = struct.unpack("<Q", read_range(url, 0, 7))[0]
            if size < 2 or size > 16 * 1024 * 1024:
                raise ValueError("unexpected safetensors header size")
            headers[shard] = (size, json.loads(read_range(url, 8, 8 + size - 1)))
        size, header = headers[shard]
        info = header[name]
        dtype = dtype_map[info["dtype"]]
        start, end = info["data_offsets"]
        expected_bytes = math.prod(info["shape"]) * torch.empty((), dtype=dtype).element_size()
        if start < 0 or end - start != expected_bytes or expected_bytes > 512 * 1024 * 1024:
            raise ValueError("invalid or oversized tensor metadata")
        print(f"Fetching {name}: {expected_bytes / 2**20:.2f} MiB", flush=True)
        payload = read_range(url, 8 + size + start, 8 + size + end - 1)
        records[name] = dict(
            shard=shard,
            dtype=info["dtype"],
            shape=info["shape"],
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        tensors[name] = torch.frombuffer(payload, dtype=dtype).reshape(info["shape"]).clone()
    cache.mkdir(parents=True, exist_ok=True)
    temporary = weights_path.with_suffix(".tmp")
    save_file(tensors, str(temporary))
    temporary.replace(weights_path)
    manifest = dict(
        model=model_id,
        revision=revision,
        tensors=records,
        tensor_bytes=sum(r["bytes"] for r in records.values()),
        cache_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return tensors, manifest
