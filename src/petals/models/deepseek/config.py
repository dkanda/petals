import os
from typing import Optional, Union

from hivemind import get_logger
from petals.models.deepseek.configuration_deepseek import DeepseekV3Config
from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig

logger = get_logger(__name__)


class DistributedDeepseekConfig(DeepseekV3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "deepseek_v3"

    # Defaults for ClientConfig that might be missing if not passed in __init__
    tuning_mode = None

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        
        # Patch for Petals compatibility: clear quantization_config if present to avoid loading errors
        # on clients that don't support the model's native quantization (e.g. fp8)
        if hasattr(config, "quantization_config"):
            del config.quantization_config
            
        return result
