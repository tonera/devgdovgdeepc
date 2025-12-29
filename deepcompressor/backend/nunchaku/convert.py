"""Program entrance for Nunchaku backend: converting a DeepCompressor state dict to a Nunchaku state dict."""

import argparse
import os

import safetensors.torch
import torch
import tqdm

from .z_image import convert_to_nunchaku_z_image_state_dicts
from .flux import convert_to_nunchaku_flux_state_dicts


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
            safetensors.torch.save_file(merged, output_file)
            print(f"Quantized model saved to {output_file}.")
        else:
            safetensors.torch.save_file(
                converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors")
            )
            safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
            print(f"Quantized model saved to {output_dirpath}.")
