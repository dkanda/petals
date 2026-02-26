from typing import Optional, Union, Sequence
import math
import psutil

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel
from hivemind.utils.logging import get_logger

from petals.models.mixtral.block import WrappedMixtralBlock
from petals.utils.convert_block import QuantType
from petals.utils.misc import get_size_in_bytes

logger = get_logger(__name__)


def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it using BloomConfig. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    if config.torch_dtype not in ("auto", None, torch.float32):
        # If config specifies float32, we override it to the default dtype below
        return config.torch_dtype
    return torch.bfloat16


def get_block_size(
    config: PretrainedConfig,
    location: str,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    quant_type: QuantType = QuantType.NONE,
    eps: float = 0.01,  # eps accounts for ~1% of metainfo for tensor descriptions, quantization tables, etc.
) -> int:
    if location == "memory":
        assert (
            dtype is not None and quant_type is not None
        ), 'get_block_size(..., location="memory") requires to specify dtype and quant_type for calculations'

    with init_empty_weights(include_buffers=False):
        block = get_model_block(config)
        n_params = sum(param.numel() for param in block.parameters())

    if location == "memory":
        if quant_type == QuantType.NONE:
            dtype = resolve_block_dtype(config, dtype)
            bytes_per_value = get_size_in_bytes(dtype)
        elif quant_type == QuantType.INT8:
            bytes_per_value = 1
        elif quant_type == QuantType.NF4:
            bytes_per_value = 4.25 / 8  # Bitness of NF4 with this config (measured empirically)
        else:
            raise ValueError(f"Unsupported quant_type={quant_type}")
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)

    return round(n_params * bytes_per_value * (1 + eps))


def get_model_block(config, layer_idx: int = 0):
    """
    The function to create a model block based on the block class
    kwargs argument **only** is necessary for specific classes, like Mixtral.
    They will not be passed to other block constructors.
    """
    if config.block_class == WrappedMixtralBlock:
        config = PreTrainedModel._autoset_attn_implementation(config)
        return config.block_class(config, layer_idx)
    return config.block_class(config)


def estimate_num_blocks(
    block_config: PretrainedConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    attn_cache_tokens: int,
    adapters: Sequence[str] = (),
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> int:
    # Logic extracted from Server._choose_num_blocks
    assert device.type in ("cuda", "mps"), (
        "GPU is not available. If you want to run a CPU-only server, please specify --num_blocks. "
        "CPU-only servers in the public swarm are discouraged since they are much slower"
    )
    num_devices = len(tensor_parallel_devices) if tensor_parallel_devices else 1

    if num_devices > 1:
        assert device.type == "cuda", f"Tensor parallelism is not supported on {device.type.upper()}"
        memory_per_device = tuple(
            torch.cuda.get_device_properties(device).total_memory for device in tensor_parallel_devices
        )
        total_memory = min(memory_per_device) * num_devices
        if max(memory_per_device) / min(memory_per_device) > 1.5:
            raise ValueError(
                "GPU devices have highly uneven memory, which makes tensor parallelism inefficient. "
                "Please launch individual servers on each GPU or set --num_blocks manually to "
                "override this exception."
            )
    elif device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
    else:
        total_memory = psutil.virtual_memory().total

    gib = 1024**3
    # Estimate of GPU memory used in rpc_backward (2 GiB for BLOOM, proportional for other models)
    autograd_memory = 2 * gib * num_devices / 14336 * block_config.hidden_size

    # Calculate cache bytes per block
    cache_values_per_block = 2 * block_config.hidden_size * attn_cache_tokens
    cache_values_per_block //= block_config.num_key_value_groups
    cache_bytes_per_block = cache_values_per_block * get_size_in_bytes(torch_dtype)

    block_size = get_block_size(block_config, "memory", dtype=torch_dtype, quant_type=quant_type)
    total_memory_per_block = block_size + cache_bytes_per_block
    if adapters:
        # Delay import of petals.utils.peft to avoid unnecessary import of bitsandbytes
        from petals.utils.peft import estimate_adapter_memory_per_block

        total_memory_per_block += estimate_adapter_memory_per_block(
            block_config,
            torch_dtype,
            adapters,
            token=token,
            cache_dir=cache_dir,
            max_disk_space=max_disk_space,
        )

    num_blocks = math.floor((total_memory - autograd_memory) / total_memory_per_block)
    if num_blocks < 1:
        logger.error(
            f"Your GPU does not have enough memory to serve at least one block. "
            f"Available memory: {total_memory / gib:.2f} GiB, "
            f"estimated memory per block: {total_memory_per_block / gib:.2f} GiB, "
            f"autograd overhead: {autograd_memory / gib:.2f} GiB. "
            f"Please decrease --attn_cache_tokens or make other changes to free up your GPU memory."
        )
        raise ValueError("Not enough GPU memory to serve at least one block")

    num_blocks = min(num_blocks, block_config.num_hidden_layers)
    logger.info(
        f"Server will fill your GPU memory with {num_blocks} transformer blocks. "
        f"If you want to leave some free GPU memory, please specify a lesser --num_blocks manually"
    )
    return num_blocks
