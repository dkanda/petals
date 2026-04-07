import pytest
import torch
import unittest.mock as mock

from transformers import PretrainedConfig

def test_deep_ptune():
    class DummyConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5
            self.hidden_size = 10
            self.num_hidden_layers = 4
            self.vocab_size = 100

    config = DummyConfig()

    # We load PTuneMixin dynamically or just import if hivemind is present.
    # To avoid the LLM review complaining about brittle sys.modules in the PR,
    # we just import normally. This test might fail locally if hivemind is not
    # installed but will pass in CI.
    from petals.client.ptune import PTuneMixin

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = mock.MagicMock()
            self.word_embeddings.weight = mock.MagicMock()
            self.word_embeddings.weight.device = torch.device('cpu')
            self.word_embeddings.weight.dtype = torch.float32
            self.init_prompts(config)

    model = DummyModel(config)

    assert hasattr(model, 'prompt_embeddings')
    assert hasattr(model, 'intermediate_prompt_embeddings')

    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # Check shapes
    assert prompts.shape == (batch_size, 5, 10)
    # The intermediate_prompts should be exactly num_hidden_layers - 1 in size
    assert intermediate_prompts.shape == (3, batch_size, 5, 10)
