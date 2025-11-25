"""InternVL3 model loading and QLoRA configuration."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# Model configurations
INTERNVL3_CONFIGS = {
    "InternVL3-8B": {
        "model_id": "OpenGVLab/InternVL3-8B",
        "vision_model": "InternViT-300M-448px",
        "llm": "internlm2_5-7b-chat",
        "max_frames": 16,
        "image_size": 448,
    },
    "InternVL3-2B": {
        "model_id": "OpenGVLab/InternVL3-2B",
        "vision_model": "InternViT-300M-448px",
        "llm": "internlm2_5-1_8b-chat",
        "max_frames": 16,
        "image_size": 448,
    },
    "InternVL3-1B": {
        "model_id": "OpenGVLab/InternVL3-1B",
        "vision_model": "InternViT-300M-448px",
        "llm": "InternLM2-1_8B",
        "max_frames": 16,
        "image_size": 448,
    },
}


def get_qlora_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Get BitsAndBytes config for 4-bit quantization.

    Args:
        load_in_4bit: Use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype (bfloat16 recommended)
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        bnb_4bit_use_double_quant: Use double quantization

    Returns:
        BitsAndBytesConfig for model loading
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def setup_qlora(
    model,
    target_modules: Optional[list] = None,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    modules_to_save: Optional[list] = None,
) -> Any:
    """
    Apply LoRA adapters to model for efficient fine-tuning.

    Args:
        model: Base model to adapt
        target_modules: Modules to apply LoRA to (auto-detected if None)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout rate for LoRA layers
        modules_to_save: Additional modules to train (not LoRA)

    Returns:
        Model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Default target modules for InternVL3 (LLM attention + MLP)
    if target_modules is None:
        target_modules = [
            # LLM attention
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # LLM MLP
            "gate_proj",
            "up_proj",
            "down_proj",
            # Vision-language connector (optional)
            # "mlp1",  # Include if you want to adapt the connector
        ]

    # Default modules to save (embedding layers)
    if modules_to_save is None:
        modules_to_save = []

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA parameters: {trainable_params:,} trainable / {all_params:,} total "
        f"({100 * trainable_params / all_params:.2f}%)"
    )

    return model


def load_internvl3(
    model_name: str = "InternVL3-8B",
    model_path: Optional[str] = None,
    use_qlora: bool = True,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
    lora_config: Optional[Dict] = None,
    max_memory: Optional[Dict] = None,  # e.g., {0: "120GiB"} for DGX Spark
) -> Tuple[Any, Any]:
    """
    Load InternVL3 model with optional QLoRA.

    Args:
        model_name: Model name from INTERNVL3_CONFIGS or custom path
        model_path: Override model path (for local models)
        use_qlora: Apply QLoRA for memory-efficient training
        device_map: Device mapping strategy
        torch_dtype: Model dtype (bfloat16 recommended)
        trust_remote_code: Trust remote code from HuggingFace
        attn_implementation: Attention implementation (flash_attention_2)
        gradient_checkpointing: Enable gradient checkpointing
        lora_config: Custom LoRA configuration dict
        max_memory: Override memory detection, e.g., {0: "120GiB"} for DGX Spark

    Returns:
        Tuple of (model, tokenizer)
    """
    # Get model config
    if model_name in INTERNVL3_CONFIGS:
        config = INTERNVL3_CONFIGS[model_name]
        model_id = model_path or config["model_id"]
    else:
        model_id = model_path or model_name
        config = {"model_id": model_id}

    logger.info(f"Loading model: {model_id}")

    # Set dtype
    dtype = getattr(torch, torch_dtype)

    # Quantization config for QLoRA
    quantization_config = None
    if use_qlora:
        quantization_config = get_qlora_config()
        logger.info("Using 4-bit quantization (QLoRA)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=False,  # InternVL uses slow tokenizer
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # Override memory detection (needed for DGX Spark unified memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    # Try to use flash attention
    try:
        model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModel.from_pretrained(model_id, **model_kwargs)
        logger.info(f"Loaded model with {attn_implementation}")
    except Exception as e:
        logger.warning(f"Flash attention not available: {e}")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(model_id, **model_kwargs)

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Apply LoRA if using QLoRA
    if use_qlora:
        lora_kwargs = lora_config or {}
        model = setup_qlora(model, **lora_kwargs)

    model.img_context_token_id = 151667
    
    # InternVL3 has built-in image transform
    # Access via model.img_context_token_id for special tokens
    return model, tokenizer


def save_lora_weights(
    model,
    output_dir: str,
    save_full_model: bool = False,
):
    """
    Save LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        output_dir: Directory to save weights
        save_full_model: Also save full merged model
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights
    model.save_pretrained(output_dir / "adapter")
    logger.info(f"Saved LoRA adapter to {output_dir / 'adapter'}")

    # Optionally merge and save full model
    if save_full_model:
        logger.info("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir / "merged")
        logger.info(f"Saved merged model to {output_dir / 'merged'}")


def load_lora_weights(
    base_model_name: str = "InternVL3-8B",
    adapter_path: str = None,
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Load base model with trained LoRA adapter.

    Args:
        base_model_name: Base model name
        adapter_path: Path to saved adapter
        **kwargs: Additional arguments for load_internvl3

    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import PeftModel

    # Load base model without LoRA
    model, tokenizer = load_internvl3(
        model_name=base_model_name,
        use_qlora=False,  # Load base model
        **kwargs,
    )

    # Load adapter
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")

    return model, tokenizer


# Memory estimation utilities
def estimate_memory_usage(
    model_name: str = "InternVL3-8B",
    batch_size: int = 4,
    max_frames: int = 16,
    sequence_length: int = 2048,
    use_qlora: bool = True,
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """
    Estimate VRAM usage for training configuration.

    Args:
        model_name: Model name
        batch_size: Batch size
        max_frames: Frames per sample
        sequence_length: Max sequence length
        use_qlora: Using QLoRA
        gradient_checkpointing: Using gradient checkpointing

    Returns:
        Dict with estimated memory usage in GB
    """
    # Base model parameters
    if "8B" in model_name:
        base_params = 8e9
        vision_params = 0.3e9  # InternViT-300M
    elif "2B" in model_name:
        base_params = 2e9
        vision_params = 0.3e9
    elif "1B" in model_name:
        base_params = 1e9
        vision_params = 0.3e9
    else:
        base_params = 8e9  # Default to 8B
        vision_params = 0.3e9

    total_params = base_params + vision_params

    # Memory calculations
    if use_qlora:
        # 4-bit: ~0.5 bytes per param
        model_memory = total_params * 0.5 / 1e9  # GB
        # LoRA adds ~0.5% trainable params
        lora_memory = total_params * 0.005 * 4 / 1e9  # FP32 for LoRA
    else:
        # FP16: 2 bytes per param
        model_memory = total_params * 2 / 1e9
        lora_memory = 0

    # Optimizer states (AdamW: 2 FP32 copies for trainable params)
    trainable = total_params * 0.005 if use_qlora else total_params
    optimizer_memory = trainable * 8 / 1e9  # 8 bytes for AdamW states

    # Gradient memory
    gradient_memory = trainable * 4 / 1e9  # FP32 gradients

    # Activation memory (rough estimate)
    # With gradient checkpointing, activations are recomputed
    if gradient_checkpointing:
        activation_factor = 0.3  # Much smaller with checkpointing
    else:
        activation_factor = 1.0

    # Per-sample activation estimate
    # Images: batch_size * max_frames * 448 * 448 * 3 * 4 bytes
    image_activations = batch_size * max_frames * 448 * 448 * 3 * 4 / 1e9

    # Sequence activations: batch_size * seq_len * hidden_dim * 4 bytes
    hidden_dim = 4096 if "8B" in model_name else 2048
    seq_activations = batch_size * sequence_length * hidden_dim * 4 / 1e9

    activation_memory = (image_activations + seq_activations) * activation_factor

    total_memory = (
        model_memory
        + lora_memory
        + optimizer_memory
        + gradient_memory
        + activation_memory
    )

    return {
        "model_memory_gb": round(model_memory, 2),
        "lora_memory_gb": round(lora_memory, 2),
        "optimizer_memory_gb": round(optimizer_memory, 2),
        "gradient_memory_gb": round(gradient_memory, 2),
        "activation_memory_gb": round(activation_memory, 2),
        "total_estimated_gb": round(total_memory, 2),
        "batch_size": batch_size,
        "use_qlora": use_qlora,
        "gradient_checkpointing": gradient_checkpointing,
    }
