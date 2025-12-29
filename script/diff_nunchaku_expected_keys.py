#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff expected keys (as built by nunchaku's ZImage transformer) vs a merged safetensors checkpoint.

This pinpoints which *non-wcscales* key is missing and causes:
    AssertionError at nunchaku.models.transformers.utils.patch_scale_key (assert ".wcscales" in k)

Usage:
  python script/diff_nunchaku_expected_keys.py --model /path/to/model.safetensors
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.models.transformers.utils import patch_scale_key
from nunchaku.utils import get_precision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Merged .safetensors path")
    ap.add_argument("--device", default="cpu", help='device for to_empty(), e.g. "cpu" or "cuda:0"')
    ap.add_argument("--torch-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    path = os.path.abspath(os.path.expanduser(args.model))
    assert os.path.isfile(path), f"File not found: {path}"

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.torch_dtype]

    # Use nunchaku's own loader to ensure the same key naming / metadata parsing.
    transformer_meta, model_state_dict, metadata = NunchakuZImageTransformer2DModel._build_model(  # type: ignore[attr-defined]
        path, torch_dtype=torch_dtype
    )
    qcfg = json.loads((metadata or {}).get("quantization_config", "{}"))
    rank = int(qcfg.get("rank", 32))
    skip_refiners = bool(qcfg.get("skip_refiners", False))

    precision = get_precision()
    if str(precision).lower() == "fp4":
        precision = "nvfp4"

    print("[diff] model =", path)
    print("[diff] quantization_config =", qcfg)
    print(f"[diff] rank={rank} skip_refiners={skip_refiners} precision={precision} torch_dtype={torch_dtype}")

    # Patch the model (this changes which modules exist => changes expected keys).
    transformer_meta = transformer_meta.to(torch_dtype)
    transformer_meta._patch_model(skip_refiners=skip_refiners, precision=precision, rank=rank)  # type: ignore[attr-defined]
    transformer_meta = transformer_meta.to_empty(device=args.device)

    expected = set(transformer_meta.state_dict().keys())
    got = set(model_state_dict.keys())
    missing = sorted(expected - got)
    extra = sorted(got - expected)

    print()
    print(f"[diff] expected={len(expected)} got={len(got)} missing={len(missing)} extra={len(extra)}")

    # Show the first missing key that would violate patch_scale_key's assumption.
    first_bad = next((k for k in missing if ".wcscales" not in k), None)
    if first_bad is not None:
        print("[diff] FIRST missing non-wcscales key (this triggers assertion):")
        print("  ", first_bad)
    else:
        print("[diff] missing keys are all wcscales-only (should be ok for patch_scale_key).")

    if missing:
        print()
        print(f"[diff] missing (first {args.limit}):")
        for k in missing[: args.limit]:
            print("  MISSING:", k)
        if len(missing) > args.limit:
            print(f"  ... and {len(missing) - args.limit} more")

    if extra:
        print()
        print(f"[diff] extra (first {args.limit}):")
        for k in extra[: args.limit]:
            print("  EXTRA  :", k)
        if len(extra) > args.limit:
            print(f"  ... and {len(extra) - args.limit} more")

    # Try to run patch_scale_key to reproduce the exact failure point.
    print()
    try:
        patch_scale_key(transformer_meta, model_state_dict)
        print("[diff] patch_scale_key: OK")
    except AssertionError as e:
        print("[diff] patch_scale_key: FAILED with AssertionError")
        if first_bad is not None:
            print("[diff] likely missing key:", first_bad)
        raise
    except Exception as e:
        print("[diff] patch_scale_key: FAILED with exception:", repr(e))
        raise


if __name__ == "__main__":
    main()


