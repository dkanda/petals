import pytest
import torch
import torch.nn as nn
import dataclasses
from typing import Optional

try:
    from petals.client.ptune import PTuneMixin
    from petals.utils.misc import DUMMY
except ImportError:
    pytest.skip("Missing dependencies for petals imports (e.g. hivemind)", allow_module_level=True)


@dataclasses.dataclass
class PretrainedConfig:
    num_hidden_layers: int = 12
    hidden_size: int = 768
    tuning_mode: str = "deep_ptune"
    pre_seq_len: int = 10


class DummyModel(PTuneMixin):
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size, dtype=torch.float32)
        # Note: In isolated testing environments, `force_non_empty_weights` inside `init_prompts`
        # requires standard PyTorch behavior. We mock or just test the init_prompts functionality.
        self.init_prompts(config)


def test_deep_ptune_intermediate_prompts_shape():
    """
    Tests that PTuneMixin properly sizes the intermediate_prompts to
    (num_hidden_layers - 1) instead of num_hidden_layers when deep_ptune is enabled.
    """
    config = PretrainedConfig(
        num_hidden_layers=12,
        hidden_size=768,
        tuning_mode="deep_ptune",
        pre_seq_len=10
    )
    model = DummyModel(config)

    batch_size = 4
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    expected_intermediate_shape = (
        config.num_hidden_layers - 1,
        batch_size,
        config.pre_seq_len,
        config.hidden_size
    )

    assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
    assert intermediate_prompts.shape == expected_intermediate_shape
