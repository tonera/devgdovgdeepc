#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a merged Nunchaku safetensors file produced by `deepcompressor.backend.nunchaku.convert`.

Goal: help debug Nunchaku loader failures like `assert ".wcscales" in k` by showing:
- metadata keys + decoded quantization_config
- counts and examples of keys ending with: qweight / wscales / wcscales / wtscale / wzeros / bias
- per-block (or per-prefix) summary of which scale keys exist

This script intentionally uses `safetensors.safe_open` (no torch required) and does not modify files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

from safetensors import safe_open


def _short(s: str, n: int = 160) -> str:
    s = s or ""
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


def _group_prefix(key: str) -> str:
    """
    Best-effort grouping: keep first 2 segments for z-image blocks: e.g. 'layers.0', 'noise_refiner.0'.
    Fallback to first segment.
    """
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] in ("layers", "noise_refiner", "context_refiner"):
        return ".".join(parts[:2])
    return parts[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to merged .safetensors file")
    ap.add_argument("--limit", type=int, default=30, help="Max example keys printed per category")
    ap.add_argument(
        "--grep",
        type=str,
        default="",
        help="Optional regex to filter keys shown in the detailed table (e.g. 'wcscales|wscales').",
    )
    args = ap.parse_args()

    path = os.path.abspath(os.path.expanduser(args.model))
    assert os.path.isfile(path), f"File not found: {path}"

    print(f"[inspect] file = {path}")
    print()

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        print("[inspect] metadata keys:", sorted(meta.keys()))
        if "model_class" in meta:
            print("[inspect] metadata.model_class:", meta.get("model_class"))
        if "quantization_config" in meta:
            qc_raw = meta.get("quantization_config", "")
            print("[inspect] metadata.quantization_config(raw):", _short(qc_raw))
            try:
                qc = json.loads(qc_raw)
                print("[inspect] metadata.quantization_config(decoded):", qc)
            except Exception as e:
                print("[inspect] metadata.quantization_config decode failed:", repr(e))
        if "config" in meta:
            print("[inspect] metadata.config(raw):", _short(meta.get("config", ""), n=240))
        print()

        keys = list(f.keys())

        # categorize by suffix token (last segment)
        suffixes = ("qweight", "wscales", "wcscales", "wtscale", "wzeros", "bias")
        by_suffix: dict[str, list[str]] = {s: [] for s in suffixes}
        others: list[str] = []
        for k in keys:
            last = k.split(".")[-1]
            if last in by_suffix:
                by_suffix[last].append(k)
            else:
                others.append(k)

        print("[inspect] tensor key counts:")
        cnt = {k: len(v) for k, v in by_suffix.items()}
        for s in suffixes:
            print(f"  - {s:8s}: {cnt[s]}")
        print(f"  - {'(other)':8s}: {len(others)}")
        print()

        # show example keys
        for s in suffixes:
            ex = sorted(by_suffix[s])[: max(0, int(args.limit))]
            if not ex:
                continue
            print(f"[inspect] examples: *.{s} (showing {len(ex)})")
            for k in ex:
                info = f.get_tensor(k)
                print(f"  - {k} :: shape={tuple(info.shape)} dtype={str(info.dtype)}")
            print()

        # look for suspicious scale keys that might trip patch_scale_key
        suspicious = sorted(by_suffix["wscales"] + by_suffix["wtscale"])
        if suspicious:
            print("[inspect] WARNING: found scale-like keys not ending with .wcscales (might trip nunchaku patch_scale_key):")
            for k in suspicious[: max(0, int(args.limit))]:
                print("  -", k)
            if len(suspicious) > int(args.limit):
                print(f"  ... and {len(suspicious) - int(args.limit)} more")
            print()

        # per-prefix summary for scales
        prefix_summary = defaultdict(Counter)  # prefix -> Counter(scale_kind)
        scale_kinds = ("wscales", "wcscales", "wtscale")
        for kind in scale_kinds:
            for k in by_suffix[kind]:
                prefix_summary[_group_prefix(k)][kind] += 1

        print("[inspect] per-prefix scale summary (prefix -> counts):")
        for p in sorted(prefix_summary.keys()):
            c = prefix_summary[p]
            print(f"  - {p:20s} :: wcscales={c.get('wcscales',0)} wscales={c.get('wscales',0)} wtscale={c.get('wtscale',0)}")
        print()

        # optional filtered key dump
        if args.grep:
            rx = re.compile(args.grep)
            matched = [k for k in keys if rx.search(k)]
            print(f"[inspect] grep={args.grep!r} matched {len(matched)} keys (showing up to {args.limit}):")
            for k in sorted(matched)[: max(0, int(args.limit))]:
                t = f.get_tensor(k)
                print(f"  - {k} :: shape={tuple(t.shape)} dtype={str(t.dtype)}")
            if len(matched) > int(args.limit):
                print(f"  ... and {len(matched) - int(args.limit)} more")
            print()


if __name__ == "__main__":
    main()


