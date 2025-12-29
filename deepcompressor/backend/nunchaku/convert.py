"""Program entrance for Nunchaku backend: converting a DeepCompressor state dict to a Nunchaku state dict."""

import argparse
import json
import os
import typing as tp

import safetensors.torch
import torch
import tqdm

from .z_image import convert_to_nunchaku_z_image_state_dicts
from .flux import convert_to_nunchaku_flux_state_dicts


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return tp.cast(dict, json.load(f))


def _find_zimage_transformer_config_json(*, diffusers_dir: str) -> dict:
    """Try best-effort to load Z-Image transformer config.json from a diffusers directory."""
    direct = os.path.join(diffusers_dir, "transformer", "config.json")
    if os.path.exists(direct):
        try:
            return _read_json(direct)
        except Exception:
            pass
    for root, _, files in os.walk(diffusers_dir):
        if "config.json" not in files:
            continue
        p = os.path.join(root, "config.json")
        try:
            cfg = _read_json(p)
            cls = str(cfg.get("_class_name", ""))
            if "ZImageTransformer" in cls or "ZImageTransformer2DModel" in cls:
                return cfg
        except Exception:
            continue
    return {}


def _infer_rank_from_branch(branch_dict: dict) -> int:
    """Infer LoRA/SVD rank from branch dict, best-effort. Returns 0 if unknown."""
    # DeepCompressor stores branch per module: { "...": {"a.weight": Tensor[r, in], "b.weight": Tensor[out, r]} }
    for _, v in branch_dict.items():
        if not isinstance(v, dict):
            continue
        a = v.get("a.weight", None)
        b = v.get("b.weight", None)
        if torch.is_tensor(a) and getattr(a, "ndim", 0) == 2:
            return int(a.shape[0])
        if torch.is_tensor(b) and getattr(b, "ndim", 0) == 2:
            return int(b.shape[1])
    return 0


def _build_zimage_metadata(*, diffusers_dir: str, flavor: str, rank: int, skip_refiners: bool) -> dict[str, str]:
    """Build nunchaku metadata for a merged Z-Image safetensors file."""
    cfg = _find_zimage_transformer_config_json(diffusers_dir=diffusers_dir) if diffusers_dir else {}
    if flavor == "fp4":
        quant_dtype = "fp4_e2m1_all"
        group_size = 16
        weight_scale_dtype: list[object] | None = [None, "fp8_e4m3_nan"]
        act_scale_dtype: object = "fp8_e4m3_nan"
    else:
        quant_dtype = "int4"
        group_size = 64
        weight_scale_dtype = None
        act_scale_dtype = None
    quantization_config: dict[str, tp.Any] = {
        "method": "svdquant",
        "weight": {"dtype": quant_dtype, "group_size": int(group_size), "scale_dtype": weight_scale_dtype},
        "activation": {"dtype": quant_dtype, "group_size": int(group_size), "scale_dtype": act_scale_dtype},
        "rank": int(rank),
        "skip_refiners": bool(skip_refiners),
    }
    return {
        "model_class": "NunchakuZImageTransformer2DModel",
        "config": json.dumps(cfg, ensure_ascii=False),
        "quantization_config": json.dumps(quantization_config, ensure_ascii=False),
        "comfy_config": json.dumps({}, ensure_ascii=False),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='torch.load map_location target. Examples: "cpu", "cuda", "cuda:0". Default: "cpu".',
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help=(
            "Optional. If set, merge all weights into a single safetensors file at this path. "
            "Otherwise, save two files: transformer_blocks.safetensors + unquantized_layers.safetensors."
        ),
    )
    parser.add_argument(
        "--diffusers-dir",
        type=str,
        default="",
        help="Optional. Diffusers directory used to embed transformer config.json into safetensors metadata (merged output only).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Optional. Rank written into metadata (merged output only). If not provided, will try to infer from branch.pt.",
    )
    parser.add_argument("--float-point", action="store_true", help="use float-point 4-bit quantization (FP4).")
    parser.add_argument("--dry-run", action="store_true", help="if set, state dicts will NOT be saved")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        # Fall back to the quant dir name (recommended: always pass --model-name).
        model_name = args.quant_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided, using {model_name} as the model name (please pass --model-name to be safe).")
    else:
        model_name = args.model_name
    assert model_name, "Model name must be provided."
    assert "flux" in model_name.lower() or "z-image" in model_name.lower(), f"{model_name} model is NOT supported so far."
    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")
    # NOTE: checkpoints may have been saved on multi-GPU machines (e.g. cuda:4). Default to CPU to be portable.
    map_location = args.device
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}
    if "flux" in model_name.lower():
        converted_state_dict, other_state_dict = convert_to_nunchaku_flux_state_dicts(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=args.float_point,
        )
    elif "z-image" in model_name.lower():
        if any("refiner" in k for k in smooth_dict):
            skip_refiners = False
        else:
            skip_refiners = True
        print(f"    - skip_refiners = {skip_refiners}")
        converted_state_dict, other_state_dict = convert_to_nunchaku_z_image_state_dicts(
            model_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=args.float_point,
            skip_refiners=skip_refiners,
        )
    else:
        raise ValueError(f"{model_name} model is NOT supported so far.")

    if args.dry_run:
        print(f"Program in DRY RUN mode. Quantized checkpoints NOT saved.")
    else:
        output_dirpath = os.path.join(args.output_root, model_name)
        os.makedirs(output_dirpath, exist_ok=True)
        if args.output_file:
            merged = {}
            merged.update(converted_state_dict)
            merged.update(other_state_dict)
            output_file = os.path.abspath(os.path.expanduser(args.output_file))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"Removed existing output file: {output_file}")
            metadata: dict[str, str] | None = None
            if "z-image" in model_name.lower():
                diffusers_dir = _abs(args.diffusers_dir) if str(args.diffusers_dir).strip() else ""
                flavor = "fp4" if bool(args.float_point) else "int4"
                rank = int(args.rank)
                if rank <= 0:
                    inferred = _infer_rank_from_branch(branch_dict)
                    rank = int(inferred) if inferred > 0 else 32
                metadata = _build_zimage_metadata(
                    diffusers_dir=diffusers_dir if diffusers_dir and os.path.isdir(diffusers_dir) else "",
                    flavor=flavor,
                    rank=rank,
                    skip_refiners=bool(skip_refiners) if "skip_refiners" in locals() else True,
                )
            safetensors.torch.save_file(merged, output_file, metadata=metadata or {})
            print(f"Quantized model saved to {output_file}.")
        else:
            safetensors.torch.save_file(
                converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors")
            )
            safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
            print(f"Quantized model saved to {output_dirpath}.")
