from petals.models.gemma3.config import DistributedGemma3Config
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma3Config,
)
