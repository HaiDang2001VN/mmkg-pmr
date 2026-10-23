"""Encoder parity gate: default encoder vs a manual dense SDPA loop.

Compares the default `nn.TransformerEncoder` against a manual dense
reference under four configurations (nested tensor on/off, MHA fast path
on/off) and reports each difference. Production requires a zero-diff
configuration; the manual loop is the dense reference.
"""

import json
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

    variants = {}
    with torch.no_grad():
        man = manual_encoder(enc, x, mask)
        variants["default"] = enc(x, src_key_padding_mask=mask)

        enc.enable_nested_tensor = False
        variants["no_nested"] = enc(x, src_key_padding_mask=mask)

        try:
            torch.backends.mha.set_fastpath_enabled(False)
            variants["no_fastpath"] = enc(x, src_key_padding_mask=mask)
            enc.enable_nested_tensor = True
            variants["no_fastpath_default_nested"] = enc(
                x, src_key_padding_mask=mask)
        except Exception as exc:  # noqa: BLE001
            variants["fastpath_error"] = str(exc)
        finally:
            try:
                torch.backends.mha.set_fastpath_enabled(True)
            except Exception:  # noqa: BLE001
                pass

    report = {}
    for name, out in variants.items():
        if torch.is_tensor(out):
            report[name] = round(float((out - man).abs().max()), 8)
        else:
            report[name] = out
    print("ACCEPTANCE encoder-parity " + json.dumps(report))
    best = min((v for v in report.values() if isinstance(v, float)),
               default=None)
    if best is None or best > 1e-5:
        print("ACCEPTANCE encoder-parity FAILED")
        sys.exit(2)
    print("ACCEPTANCE encoder-parity PASSED")


if __name__ == "__main__":
    main()
