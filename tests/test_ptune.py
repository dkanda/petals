import pytest
import torch
from transformers import PretrainedConfig

try:
    from petals.client.ptune import PTuneMixin
    import hivemind
except ImportError:
    pytest.skip("Missing dependencies (e.g. hivemind) for ptune tests", allow_module_level=True)

class DummyModel(PTuneMixin):
    def __init__(self, config):
        self.config = config
        self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_ptune_mixin_intermediate_prompts_shape():
    """
    Test that the intermediate_prompts shape is correct when using deep_ptune,
    specifically that it uses (num_hidden_layers - 1).
    """
    batch_size = 2
    pre_seq_len = 4
    hidden_size = 8
    num_hidden_layers = 3

    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # prompts should be of shape (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)

    # intermediate_prompts should be of shape (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)

def test_ptune_mixin_ptune_mode():
    """
    Test ptune mode which should not have intermediate prompts.
    """
    batch_size = 2
    pre_seq_len = 4
    hidden_size = 8
    num_hidden_layers = 3

    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=pre_seq_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    assert prompts.shape == (batch_size, pre_seq_len, hidden_size)

    from petals.utils.misc import DUMMY
    assert intermediate_prompts is DUMMY
