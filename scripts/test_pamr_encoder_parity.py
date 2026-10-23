"""Encoder parity gate: default nn.TransformerEncoder vs a manual SDPA loop.

Run on CPU. Exits 0 only when the two implementations agree within 1e-5 on
realistic padding masks; this guards the evaluation path against future
PyTorch fast-path regressions.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_encoder(encoder, src, key_padding_mask):
    x = src
    for layer in encoder.layers:
        mha = layer.self_attn
        heads = mha.num_heads
        emb = mha.embed_dim
        head_dim = emb // heads
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(t):
            return t.view(t.shape[0], t.shape[1], heads, head_dim).permute(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        am = None
        if key_padding_mask is not None:
            am = torch.zeros(key_padding_mask.shape[0], 1, 1,
                             key_padding_mask.shape[1], device=x.device,
                             dtype=x.dtype)
            am = am.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        attn = attn.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], emb)
        attn = mha.out_proj(attn)
        x = layer.norm1(x + layer.dropout1(attn))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(ff))
    return x


def main():
    torch.manual_seed(0)
    d_model, heads, layers = 64, 4, 3
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, 0.0,
                                   activation="gelu", batch_first=True),
        num_layers=layers)
    enc.eval()
    x = torch.randn(4, 7, d_model)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[0, 5:] = True
    mask[2, 6:] = True
    mask[3, 3:] = True
    with torch.no_grad():
        out = enc(x, src_key_padding_mask=mask)
        man = manual_encoder(enc, x, mask)
    diff = float((out - man).abs().max())
    print(f"ACCEPTANCE encoder-parity max_abs={diff:.8f}")
    if diff > 1e-5:
        print("ACCEPTANCE encoder-parity FAILED")
        sys.exit(2)
    print("ACCEPTANCE encoder-parity PASSED")


if __name__ == "__main__":
    main()
