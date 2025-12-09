from petals.models.deepseek.block import WrappedDeepseekBlock
from petals.models.deepseek.config import DistributedDeepseekConfig
from petals.models.deepseek.model import (
    DistributedDeepseekForCausalLM,
    DistributedDeepseekForSequenceClassification,
    DistributedDeepseekModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedDeepseekConfig,
    model=DistributedDeepseekModel,
    model_for_causal_lm=DistributedDeepseekForCausalLM,
    model_for_sequence_classification=DistributedDeepseekForSequenceClassification,
)
