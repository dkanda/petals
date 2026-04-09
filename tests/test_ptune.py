import torch
import torch.nn as nn
from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_shapes():
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = DummyModel(config)

    # Check parameters
    assert model.prompt_embeddings.weight.shape == (5, 16)
    assert model.intermediate_prompt_embeddings.weight.shape == (5, (4 - 1) * 16)

    # Check output shape
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    assert intermediate_prompts.shape == (4, 2, 5, 16)

    # Check that first layer in intermediate_prompts is padded with zeros
    assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_shapes():
    config = PretrainedConfig(
        tuning_mode="ptune",
        pre_seq_len=5,
        hidden_size=16,
        num_hidden_layers=4,
    )
    model = DummyModel(config)

    # Check parameters
    assert model.prompt_embeddings.weight.shape == (5, 16)
    assert not hasattr(model, "intermediate_prompt_embeddings")

    # Check output shape
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 16)
    from petals.utils.misc import DUMMY
    assert (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0) or intermediate_prompts is DUMMY
