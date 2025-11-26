from heliopetals.models.bloom.block import WrappedBloomBlock
from heliopetals.models.bloom.config import DistributedBloomConfig
from heliopetals.models.bloom.model import (
    DistributedBloomForCausalLM,
    DistributedBloomForSequenceClassification,
    DistributedBloomModel,
)
from heliopetals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedBloomConfig,
    model=DistributedBloomModel,
    model_for_causal_lm=DistributedBloomForCausalLM,
    model_for_sequence_classification=DistributedBloomForSequenceClassification,
)
