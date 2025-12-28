import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.gemma2 import Gemma2Config

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig

logger = get_logger(__name__)


class DistributedGemma3Config(Gemma2Config, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "gemma2"
    # This model is multimodal, but we only support text inputs for now.
    # We ignore vision-related fields to treat it as a text-only transformer.
    # To be filled in later when block execution is implemented.
    # block_class = WrappedGemma3Block
    # attn_class = Gemma3Attention
    block_prefix = "model.layers"

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        if dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        if "quantization_config" in kwargs:
            del kwargs["quantization_config"]

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        config.pretraining_tp = 1  # This may give less accurate results but it doesn't matter if we use quantization
        config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by Petals
        return result
