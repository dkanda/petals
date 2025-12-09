
from petals.models.deepseek.configuration_deepseek import DeepseekV3Config
from petals.client.config import ClientConfig


class DistributedDeepseekConfig(DeepseekV3Config, ClientConfig):
    model_type = "deepseek_v3"

    # Defaults for ClientConfig that might be missing if not passed in __init__
    tuning_mode = None
