# VLA Fine-Tuning: InternVL3-8B on RoboVQA

Fine-tuning a Vision-Language-Action model for robotics scene understanding using curriculum learning on DGX Spark (GB10 Blackwell GPU, 128GB unified memory).

## Data Preprocessing

### Source Data
The RoboVQA dataset is hosted at `gs://anon_robovqa` as TFRecords containing 221,912 video samples with 16 frames each. Each sample includes multiple QA pairs covering task types: affordance detection (33%), planning variants (35%), success prediction (14%), future prediction (11%), and past description (6%).

### Preprocessing Pipeline

We extract frames and generate separate JSONL files for each training stage:

```
data/robovqa/processed/
├── images/           # 3.5M JPEGs (16 frames × 221K samples) @ 288×288
├── stage1/
│   ├── train.jsonl   # 722,979 samples (split QAs)
│   └── val.jsonl     # 80,330 samples
├── stage2/
│   ├── train.jsonl   # 199,721 samples (multi-turn)
│   └── val.jsonl     # 22,191 samples
└── metadata.json
```

**Key design decision:** Images are stored once in a shared folder and referenced by both stages. This is disk-efficient (~62GB) and adds minimal dataloader overhead compared to duplicating images per QA pair.

The QA text structure follows the format `<task:type:subtype:format>` with questions applying to the final frame (frame 15) while all 16 frames provide temporal context.

## Model Selection

We evaluated three candidate models for video-based robotics VQA:

| Model | Video Support | DGX Spark Support | Decision |
|-------|--------------|-------------------|----------|
| PaliGemma 3B | ❌ Requires frame extraction | ❌ No official support | Rejected |
| Qwen2.5-VL-7B | ✅ Native video | ⚠️ Community only | Considered |
| InternVL3-8B | ✅ Native video | ✅ Official playbooks | **Selected** |

**Why InternVL3-8B:**
- NVIDIA provides official VLM fine-tuning playbooks for DGX Spark using InternVL3
- Native multi-image/video input matches RoboVQA's 16-frame format
- Strong embodied AI benchmarks and well-documented QLoRA fine-tuning recipes
- 8B parameters fit comfortably in 128GB unified memory with QLoRA

We later compared against Qwen3-VL (released Nov 2025) which shows better temporal grounding and spatial reasoning benchmarks. However, we completed our InternVL3 experiments first given the existing infrastructure investment.

## Two-Stage Curriculum Learning

### Motivation
RoboVQA samples contain multiple QA pairs per video. Training directly on multi-turn conversations requires the model to handle complex reasoning chains before mastering basic visual grounding—a difficult learning objective.

### Curriculum Design

**Stage 1: Visual Grounding**
- Split each sample's QA pairs into separate training examples
- One question-answer per sample, all 16 frames as context
- Teaches: scene understanding, object recognition, single-task responses
- 722,979 training samples

**Stage 2: Multi-Turn Reasoning**
- All QA pairs from one video become a single multi-turn conversation
- First turn includes image tokens; subsequent turns are text-only
- Teaches: reasoning chains, task relationships, conversational coherence
- 199,721 training samples
- Initialized from Stage 1 checkpoint

## Hyperparameters

Aligned with InternVL2/3 official LoRA fine-tuning scripts:

| Parameter | Stage 1 | Stage 2 | Notes |
|-----------|---------|---------|-------|
| LoRA rank (r) | 128 | 128 | InternVL official recommendation |
| LoRA alpha | 256 | 256 | 2×r heuristic |
| LoRA dropout | 0.05 | 0.05 | — |
| Learning rate | 4e-5 | 2e-5 | Lower for continued fine-tuning |
| Batch size | 4 | 2 | Smaller for longer sequences |
| Gradient accumulation | 8 | 16 | Effective batch size: 32 |
| Epochs | 1 | 1 | Prevent overfitting on large dataset |
| Warmup ratio | 0.03 | 0.03 | — |
| LR scheduler | Cosine | Cosine | — |
| Max grad norm | 1.0 | 1.0 | — |
| Optimizer | AdamW | AdamW | β1=0.9, β2=0.999 |
| Weight decay | 0.01 | 0.01 | — |
| Max frames | 4 | 10 | Memory vs. temporal context tradeoff |
| Max sequence length | 2048 | 4096 | Stage 2 needs longer for multi-turn |
| Attention | SDPA | SDPA | See Flash Attention section |
| Precision | bfloat16 | bfloat16 | Mixed precision training |

**LoRA target modules:** Auto-detected (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

## 10% Subset Validation

Before full-scale training, we validated the pipeline on 10% of the data (`subset_ratio: 0.1`):

### Results

| Stage | Eval Loss | Perplexity | Training Time |
|-------|-----------|------------|---------------|
| Stage 1 | 0.284 | 1.329 | ~15 hours |
| Stage 2 | 0.034 | 1.034 | ~42 hours |

**Observations:**
- Consistent improvement throughout Stage 1 with no overfitting
- Stage 2 perplexity near 1.0 indicates high confidence on multi-turn reasoning
- Eval loss decreased monotonically, suggesting the model could benefit from more data

## Curriculum Validation: Stage 2-Only Experiment

To verify that the two-stage curriculum provides value over direct multi-turn training, we ran an ablation:

**Experiment:** Train directly on Stage 2 data (multi-turn) without Stage 1 pre-training, then evaluate on both Stage 1 and Stage 2 validation sets.

### Results

| Eval Dataset | Curriculum (S1→S2) | Stage 2-Only | Δ |
|--------------|-------------------|--------------|---|
| Stage 1 (single QA) | **0.284** / 1.33 ppl | 0.495 / 1.64 ppl | +74% loss |
| Stage 2 (multi-turn) | **0.034** / 1.03 ppl | 0.043 / 1.04 ppl | +26% loss |

**Conclusion:** The Stage 1 visual grounding phase is essential. Training directly on multi-turn data without the single-QA foundation results in significantly worse performance on basic visual understanding tasks (+74% loss) and modest degradation on multi-turn reasoning (+26% loss).

The two-stage curriculum is justified.

## Flash Attention 2 on GB10 Blackwell

### The Problem
GB10 uses compute capability sm_121 (Blackwell architecture). Flash Attention 2 officially supports up to sm_90 (Hopper). PyTorch's maximum supported capability is sm_120.

### Compilation Attempt

We attempted to compile Flash Attention 2.7.2 from source with sm_121 support:

1. **Added sm_121 to setup.py:**
   ```python
   cc_flag.append("-gencode")
   cc_flag.append("arch=compute_121,code=sm_121")
   ```

2. **Updated CUTLASS to v4.3.0** for CUDA 13.0 compatibility

3. **Patched CUTLASS headers** (`cuda_host_adapter.hpp`) to add missing CUDA 13.0 type definitions:
   ```cpp
   typedef CUresult (*PFN_cuTensorMapEncodeIm2col)(...);
   typedef CUresult (*PFN_cuTensorMapEncodeTiled)(...);
   ```

4. **Patched flash_api.cpp** runtime checks to allow sm_12x devices

### Result

Compilation succeeded and kernels executed without errors. However, benchmarking revealed:

| Implementation | Latency (512 tokens) |
|----------------|---------------------|
| Flash Attention 2 | 489ms |
| Flash Attention 3 | 489ms |
| **SDPA** | **481ms** |

**SDPA is actually 2% faster than Flash Attention on GB10.**

The compiled Flash Attention kernels run but are not optimized for Blackwell's architecture (different warp scheduling, shared memory configuration, tensor core instructions). They likely fall back to generic PTX code paths.

### Decision
Use PyTorch's native SDPA (`attn_implementation: sdpa`) until official Blackwell-optimized Flash Attention kernels are released. The expected 2-4x speedup from Flash Attention is not achievable on current hardware without proper kernel tuning.

## Repository Structure

```
vla-ft/
├── vlaft/
│   ├── data/
│   │   ├── download_robovqa.py
│   │   ├── inspect_robovqa.py
│   │   └── preprocess_robovqa.py
│   ├── models/
│   │   └── internvl.py
│   └── training/
│       └── trainer.py
├── scripts/
│   ├── train.py
│   └── eval.py
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage2_only.yaml
└── experiments/
    └── {experiment_name}_{timestamp}/
        ├── checkpoints/
        ├── logs/
        │   ├── training.log
        │   ├── *_train_metrics.csv
        │   └── *_eval_metrics.csv
        └── config.yaml
```

## Hardware

- **System:** DGX Spark
- **GPU:** NVIDIA GB10 (Blackwell, sm_121)
- **Memory:** 128GB unified memory
- **CUDA:** 13.0