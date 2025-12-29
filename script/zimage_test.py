import argparse
import os
import torch
from diffusers import ZImagePipeline
from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Nunchaku 权重文件路径 或 Diffusers 模型目录")
parser.add_argument("--base", help="基座 Diffusers 模型目录 (Nunchaku 模式必需)")
parser.add_argument("--prompt", default="a cute dog")
args = parser.parse_args()

device = "cuda"
torch_dtype = torch.bfloat16

# 检测 --model 是文件还是目录
model_path = os.path.expanduser(args.model)
is_diffusers_dir = os.path.isdir(model_path)

if is_diffusers_dir:
    print(f"[INFO] 检测到 Diffusers 目录模式: {model_path}")
else:
    print(f"[INFO] 检测到 Nunchaku 文件模式: {model_path}")
    # Nunchaku 模式必须提供 --base
    if not args.base:
        parser.error("Nunchaku 文件模式需要提供 --base 参数 (基座 Diffusers 模型目录)")

if not is_diffusers_dir:
    # ---- Nunchaku 模式: 原有逻辑 ----
    
    # ---- hardware/precision fail-fast ----
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            name = torch.cuda.get_device_name(dev)
            cc = torch.cuda.get_device_capability(dev)
            print(f"[HW] cuda_device={dev} name={name} capability={cc}")
        prec = get_precision()
        print(f"[HW] nunchaku get_precision() -> {prec}")
    except Exception as e:
        print(f"[WARN] failed to query hw/precision: {e}")

    # ---- fail-fast preflight: print the FIRST missing/extra key before patch_scale_key asserts ----
    try:
        transformer_meta, model_state_dict, metadata = NunchakuZImageTransformer2DModel._build_model(  # type: ignore[attr-defined]
            args.model, torch_dtype=torch_dtype
        )
        try:
            import json as _json

            qcfg = _json.loads((metadata or {}).get("quantization_config", "{}"))
            wdt = (((qcfg.get("weight") or {}).get("dtype")) or "")
            wdt_l = str(wdt).lower()
            is_fp4_ckpt = "fp4" in wdt_l
            # IMPORTANT:
            # Use checkpoint precision to patch the model, NOT hardware precision.
            # On Blackwell, get_precision() may report "fp4" even when loading an INT4 checkpoint.
            precision_from_ckpt = "nvfp4" if is_fp4_ckpt else "int4"

            if is_fp4_ckpt:
                # On non-Blackwell GPUs, nunchaku's get_precision() is typically "int4" and fp4 kernels won't be used.
                # Loading fp4 weights under int4 kernels will produce garbage (checkerboard artifacts).
                prec = get_precision()
                if str(prec).lower() != "fp4":
                    print(
                        f"[FAIL-FAST] You are loading an FP4 checkpoint but nunchaku precision='{prec}'. "
                        "FP4 requires Blackwell (sm120/121). On other GPUs use INT4 checkpoint instead."
                    )
            # IMPORTANT: compare against the *patched* model state_dict (this matches what `from_pretrained()` loads).
            rank = int(qcfg.get("rank", 32))
            skip_refiners = bool(qcfg.get("skip_refiners", False))
            transformer_meta._patch_model(skip_refiners=skip_refiners, precision=precision_from_ckpt, rank=rank)  # type: ignore[attr-defined]
        except Exception:
            pass
        expected = set(transformer_meta.state_dict().keys())
        got = set(model_state_dict.keys())
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        # Reduce noise: for nvfp4, `.wcscales` may be missing and will be auto-filled by patch_scale_key();
        # `.wtscale` is stored as an extra scalar key and will be popped by patch_scale_key().
        miss_hard = [k for k in missing if ".wcscales" not in k]
        extra_hard = [k for k in extra if not k.endswith(".wtscale")]
        if miss_hard:
            print("[FAIL-FAST] checkpoint is missing required keys (non-wcscales). First 30:")
            for k in miss_hard[:30]:
                print("  MISSING:", k)
        elif missing:
            print(f"[INFO] missing only wcscales (ok): {len(missing)}")
        if extra_hard:
            print("[FAIL-FAST] checkpoint has extra keys (non-wtscale). First 30:")
            for k in extra_hard[:30]:
                print("  EXTRA  :", k)
        elif extra:
            print(f"[INFO] extra only wtscale (ok): {len(extra)}")

        # ---- dtype mismatch triage (helps debug patch_scale_key assertion) ----
        try:
            exp_sd = transformer_meta.state_dict()
            mism = []
            for k in sorted(set(exp_sd.keys()) & set(model_state_dict.keys())):
                a = exp_sd.get(k)
                b = model_state_dict.get(k)
                if torch.is_tensor(a) and torch.is_tensor(b) and a.dtype != b.dtype:
                    # wcscales may be optional; wtscale is popped later; keep them but label.
                    mism.append((k, str(a.dtype), str(b.dtype), tuple(a.shape), tuple(b.shape)))
                    if len(mism) >= 20:
                        break
            if mism:
                print("[FAIL-FAST] dtype mismatches between model params and checkpoint (first 20):")
                for k, adt, bdt, ash, bsh in mism:
                    print(f"  DTYPE: {k}  model={adt}{ash}  ckpt={bdt}{bsh}")
        except Exception as e:
            print(f"[WARN] dtype mismatch triage failed: {e}")
    except Exception as e:
        print(f"[WARN] preflight key diff failed: {e}")

    # 加载 Nunchaku transformer
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(args.model, torch_dtype=torch_dtype)
    pipe = ZImagePipeline.from_pretrained(
        args.base,
        transformer=transformer,
        torch_dtype=torch_dtype
    ).to(device)
    
else:
    # ---- Diffusers 目录模式: 新增逻辑 ----
    print(f"[INFO] 直接加载 Diffusers 模型: {model_path}")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    ).to(device)

image = pipe(
    prompt=args.prompt,
    guidance_scale=1,
    num_inference_steps=4,
    width=1024,
    height=1024,
    generator=torch.Generator(device).manual_seed(0)
).images[0]
image.save("/home/tonera/website/output/zimage_nunchaku_test.png")

