# Z-Image 模型量化完整指南

本文档详细介绍如何使用 DeepCompressor 对 Z-Image 模型进行 INT4 量化，包括校准数据收集、模型量化、效果验证和格式转换的完整流程。

## 目录

- [前置准备](#前置准备)
- [量化流程概览](#量化流程概览)
- [步骤 1: 收集校准数据](#步骤-1-收集校准数据)
- [步骤 2: 执行模型量化](#步骤-2-执行模型量化)
- [步骤 3: 验证量化效果](#步骤-3-验证量化效果可选)
- [步骤 4: 转换为 Nunchaku 格式](#步骤-4-转换为-nunchaku-后端格式)
- [完整流程脚本](#完整流程脚本示例)
- [配置调整指南](#配置调整指南)
- [注意事项](#注意事项)

---

## 前置准备

### 环境变量设置

根据你的 GPU 架构设置 CUDA 编译目标：

```bash
# RTX 4090: 8.9
# H100: 9.0
# A100: 8.0
export TORCH_CUDA_ARCH_LIST="9.0"  # 请根据实际 GPU 修改
```

### 目录结构准备

创建必要的工作目录：

```bash
# 校准数据目录
mkdir -p /data/dongd/dc_calib

# 量化模型保存目录
mkdir -p /data/dongd/dc_saved_model

# 转换后模型目录
mkdir -p /data/dongd/dc_converted_model
```

### 可用的模型配置文件

| 配置文件 | 说明 |
|---------|------|
| `z-image-turbo.yaml` | 基础配置，量化所有层 |
| `z-image-turbo-rank64.yaml` | 使用 rank=64 的低秩分解 |
| `z-image-turbo-rank128.yaml` | 使用 rank=128 的低秩分解 |
| `z-image-turbo-rank128-skip-refiners.yaml` | rank=128，跳过 nrk/crk 层 (推荐) |
| `z-image-turbo-skip-refiners.yaml` | 跳过 nrk/crk 层 |

---

## 量化流程概览

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. 收集校准数据  │ ──▶ │  2. 模型量化     │ ──▶ │  3. 效果验证     │ ──▶ │  4. 格式转换     │
│    (Calibration) │     │   (Quantize)    │     │   (Validate)    │     │   (Convert)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 步骤 1: 收集校准数据

校准数据用于量化过程中的范围估计和平滑参数优化。

### 命令
export TORCH_CUDA_ARCH_LIST="12.1"
```bash
python3 -m deepcompressor.app.diffusion.dataset.collect.calib \
    examples/diffusion/configs/collect/qdiff.yaml \
    examples/diffusion/configs/model/z-image-turbo_smoke.yaml 
```

### 配置说明

**模型配置 (`z-image-turbo.yaml`)：**
- `pipeline.name`: 模型名称
- `pipeline.path`: 模型路径(diffusers目录)
- `pipeline.dtype`: 模型数据类型 (torch.bfloat16)
- `collect.root`: 校准数据输出路径

**校准配置 (`qdiff.yaml`)：**
```yaml
collect:
  root: datasets
  dataset_name: qdiff
  data_path: examples/diffusion/prompts/qdiff.yaml  # 包含 1000+ 条文本提示
  num_samples: 128  # 校准样本数量
```

### 输出

校准数据保存至：`/data/dongd/dc_calib/datasets/{dtype}/{model}/{protocol}/{data}/s128`

### 后台运行

```bash
nohup python3 -m deepcompressor.app.diffusion.dataset.collect.calib \
    examples/diffusion/configs/model/z-image-turbo.yaml \
    examples/diffusion/configs/collect/qdiff.yaml \
    > z_image_turbo_calib_$(date +%Y%m%d_%H%M).log 2>&1 &
```

---

## 步骤 2: 执行模型量化

使用 SVDQuant 算法对模型进行 nvfp4 量化。

### 方式 A: 基础量化

对所有层进行量化：

```bash
python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo_smoke.yaml \
    examples/diffusion/configs/svdquant/nvfp4.yaml \
    --save-model /home/tonera/project/devgdovgdeepc/dc_saved_model/Z_IMAGE_TURBO_$(date +%Y%m%d_%H%M) \
    --copy-on-save true \
    --skip-eval true \
    --eval-benchmarks .tmp/qdiff.yaml 
```

### 方式 B: 带 Low-Rank 分解的量化 (推荐)

使用 rank=128 的低秩分解，跳过 refiners 层：

```bash
python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo-rank128-skip-refiners.yaml \
    examples/diffusion/configs/svdquant/nvfp4.yaml \
    --save-model /home/tonera/project/devgdovgdeepc/dc_saved_model/Z_IMAGE_TURBO_$(date +%Y%m%d_%H%M) \
    --copy-on-save true \
    --skip-eval true \
    --eval-benchmarks .tmp/qdiff.yaml 
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--save-model` | 量化模型的保存路径 |
| `--copy-on-save` | 保存时复制原始模型权重 |
| `--skip-eval` | 跳过完整评估流程 |

### INT4 量化配置

`examples/diffusion/configs/svdquant/int4.yaml` 内容：

```yaml
quant:
  wgts:
    dtype: sint4           # 权重 4-bit 有符号整数
    group_shapes:
    - - 1
      - 64                 # 每 64 个元素为一组
      - 1
      - 1
      - 1
    scale_dtypes:
    - null
  ipts:
    static: false          # 动态激活量化
    dtype: sint4           # 激活 4-bit 有符号整数
    group_shapes:
    - - 1
      - 64
      - 1
      - 1
      - 1
    scale_dtypes:
    - null
    allow_unsigned: true   # 允许无符号量化
pipeline:
  shift_activations: false
```

### 后台运行

```bash
nohup python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo-rank128-skip-refiners.yaml \
    examples/diffusion/configs/svdquant/int4.yaml \
    --save-model /data/dongd/dc_saved_model/Z_IMAGE_TURBO_$(date +%Y%m%d_%H%M) \
    --copy-on-save true \
    --skip-eval true \
    > z_image_turbo_quantize_$(date +%Y%m%d_%H%M).log 2>&1 &
```

---

## 步骤 3: 验证量化效果 (可选)

加载量化后的模型生成图像，验证量化质量。

### 命令

```bash
TORCH_CUDA_ARCH_LIST="9.0" python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo.yaml \
    examples/diffusion/configs/svdquant/int4.yaml \
    --load-from /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251204_0743 \
    --skip-eval true
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--load-from` | 指定之前保存的量化模型路径 |
| `--skip-eval` | 跳过完整评估，仅生成少量样例图像 |

---

## 步骤 4: 转换为 Nunchaku 后端格式

将量化模型转换为 Nunchaku 推理后端可用的格式，用于高效部署。

### 命令

```bash
python -m deepcompressor.backend.nunchaku.convert \
    --quant-path ./dc_saved_model/Z_IMAGE_TURBO_20251229_1228 \
    --output-root ./dc_converted_model \
    --model-name z-image-turbo \
    --device cuda:0 \
    --float-point \
    --output-file ./dc_converted_model/z-image-turbo/svd-fp4-Beyond_Reality_v1.safetensors \
    --diffusers-dir /home/tonera/SVDQuant/Beyond_Reality_v1/diffusers \
    --rank 32

```
检查key
```bash
python script/diff_nunchaku_expected_keys.py \
  --model ./dc_converted_model/z-image-turbo/svd-fp4-Beyond_Reality_v1-r32.safetensors \
  --device cpu \
  --torch-dtype bf16 \
  --limit 20
```
测试出图
```bash
python script/zimage_test.py \
  --model ./dc_converted_model/z-image-turbo/svd-fp4-Beyond_Reality_v1-r32.safetensors \
  --base /home/tonera/SVDQuant/Beyond_Reality_v1/diffusers \
  --prompt "a cute dog"
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--quant-path` | 量化模型的保存路径 |
| `--output-root` | 转换后模型的输出目录 |
| `--model-name` | 模型名称标识 |

### 后台运行

```bash
nohup python -m deepcompressor.backend.nunchaku.convert \
    --quant-path /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251204_0743 \
    --output-root /data/dongd/dc_converted_model/Z_IMAGE_TURBO_20251204_0743_r128 \
    --model-name z-image-turbo \
    > z_image_turbo_convert_$(date +%Y%m%d_%H%M).log 2>&1 &
```

---

## 完整流程脚本示例

创建一个自动化脚本 `quantize_z_image.sh`：

```bash
#!/bin/bash
set -e

# ==================== 配置区域 ====================
export TORCH_CUDA_ARCH_LIST="12.1"  # 根据 GPU 修改

TIMESTAMP=$(date +%Y%m%d_%H%M)
MODEL_CONFIG="examples/diffusion/configs/model/z-image-turbo-rank128-skip-refiners.yaml"
QUANT_CONFIG="examples/diffusion/configs/svdquant/int4.yaml"
CALIB_CONFIG="examples/diffusion/configs/collect/qdiff.yaml"

QUANT_OUTPUT="/data/dongd/dc_saved_model/Z_IMAGE_TURBO_${TIMESTAMP}"
CONVERT_OUTPUT="/data/dongd/dc_converted_model/Z_IMAGE_TURBO_${TIMESTAMP}_r128"
LOG_DIR="./logs"

# ==================== 初始化 ====================
mkdir -p ${LOG_DIR}

echo "============================================"
echo "Z-Image 模型量化流程"
echo "时间戳: ${TIMESTAMP}"
echo "============================================"

# ==================== Step 1: 校准 ====================
echo ""
echo "[Step 1/4] 收集校准数据..."
python3 -m deepcompressor.app.diffusion.dataset.collect.calib \
    examples/diffusion/configs/model/z-image-turbo.yaml \
    ${CALIB_CONFIG} \
    2>&1 | tee ${LOG_DIR}/calib_${TIMESTAMP}.log

if [ $? -ne 0 ]; then
    echo "❌ 校准数据收集失败！"
    exit 1
fi
echo "✅ 校准数据收集完成"

# ==================== Step 2: 量化 ====================
echo ""
echo "[Step 2/4] 执行模型量化..."
python3 -m deepcompressor.app.diffusion.ptq \
    ${MODEL_CONFIG} \
    ${QUANT_CONFIG} \
    --save-model ${QUANT_OUTPUT} \
    --copy-on-save true \
    --skip-eval true \
    2>&1 | tee ${LOG_DIR}/quantize_${TIMESTAMP}.log

if [ $? -ne 0 ]; then
    echo "❌ 模型量化失败！"
    exit 1
fi
echo "✅ 模型量化完成"

# ==================== Step 3: 验证 (可选) ====================
echo ""
echo "[Step 3/4] 验证量化效果..."
python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo.yaml \
    ${QUANT_CONFIG} \
    --load-from ${QUANT_OUTPUT} \
    --skip-eval true \
    2>&1 | tee ${LOG_DIR}/validate_${TIMESTAMP}.log

echo "✅ 量化效果验证完成"

# ==================== Step 4: 转换 ====================
echo ""
echo "[Step 4/4] 转换为 Nunchaku 格式..."
python -m deepcompressor.backend.nunchaku.convert \
    --quant-path ${QUANT_OUTPUT} \
    --output-root ${CONVERT_OUTPUT} \
    --model-name z-image-turbo \
    2>&1 | tee ${LOG_DIR}/convert_${TIMESTAMP}.log

if [ $? -ne 0 ]; then
    echo "❌ 模型转换失败！"
    exit 1
fi
echo "✅ 模型转换完成"

# ==================== 完成 ====================
echo ""
echo "============================================"
echo "🎉 量化流程全部完成！"
echo "============================================"
echo "量化模型路径: ${QUANT_OUTPUT}"
echo "转换模型路径: ${CONVERT_OUTPUT}"
echo "日志目录:     ${LOG_DIR}"
echo "============================================"
```

使用方式：

```bash
chmod +x quantize_z_image.sh
./quantize_z_image.sh
```

---

## 配置调整指南

### 1. 调整 Low-Rank 分解等级

在模型配置文件中修改 `low_rank.rank` 参数：

```yaml
quant:
  wgts:
    low_rank:
      rank: 128  # 可选值: 32, 64, 128
```

| Rank 值 | 压缩效果 | 精度影响 |
|---------|---------|---------|
| 32 | 最高压缩率 | 精度损失较大 |
| 64 | 平衡方案 | 精度损失适中 |
| 128 | 较低压缩率 | 精度损失较小 |

### 2. 跳过特定层的量化

在配置文件的 `skips` 列表中添加要跳过的层类型：

```yaml
quant:
  wgts:
    low_rank:
      skips:
        - nrk                    # 非残差块 (non-residual kernel)
        - crk                    # 条件残差块 (conditional residual kernel)
        - transformer_norm       # Transformer 归一化层
        - transformer_add_norm   # Transformer Add 归一化层
        - attn_add               # Attention Add 层
        - ffn_add                # FFN Add 层
```

### 3. 调整校准样本数

在校准配置中修改 `num_samples`：

```yaml
quant:
  calib:
    num_samples: 128   # 校准样本数
```

> 💡 样本数越多，量化精度越高，但校准时间也越长。推荐值：64-256。

### 4. 可用的量化配置

| 配置文件 | 说明 |
|---------|------|
| `int4.yaml` | INT4 量化 (默认推荐) |
| `nvfp4.yaml` | NVIDIA FP4 量化 |
| `gptq.yaml` | GPTQ 量化方案 |
| `fast.yaml` | 快速量化模式 |

---

## 注意事项

### 硬件要求

- **GPU 显存**: 建议 24GB+ 显存 (如 RTX 4090, A100, H100)
- **CPU 内存**: 建议 64GB+ 系统内存
- **存储空间**: 校准数据和模型文件需要约 50-100GB 空间

### 时间消耗

| 步骤 | 预估时间 |
|------|---------|
| 校准数据收集 | 1-2 小时 |
| 模型量化 | 2-4 小时 |
| 格式转换 | 10-30 分钟 |

> 💡 建议使用 `nohup` 或 `screen` 在后台运行长时间任务。

### 常见问题

1. **CUDA 内存不足**
   - 减小 `sample_batch_size` 和 `element_batch_size`
   - 使用更高显存的 GPU

2. **校准数据路径错误**
   - 检查 `collect.root` 配置是否正确
   - 确保路径存在且有写入权限

3. **量化精度下降明显**
   - 增加校准样本数 (`num_samples`)
   - 使用更高的 Low-Rank 等级
   - 跳过敏感层的量化

---

## 相关资源

- 模型配置目录: `examples/diffusion/configs/model/`
- 量化配置目录: `examples/diffusion/configs/svdquant/`
- 校准配置目录: `examples/diffusion/configs/collect/`
- 示例脚本目录: `z_image_scripts/`

---

## 更新日志

- **2024-12-28**: 初始版本，包含完整的 Z-Image Turbo 量化流程

