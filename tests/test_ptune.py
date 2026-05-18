import torch
import torch.nn as nn
from unittest import mock
import sys

def test_ptunemixin_deep_ptune_shape():
    mock_hivemind = mock.MagicMock()
    mock_transformers = mock.MagicMock()
    mock_misc = mock.MagicMock()
    mock_misc.DUMMY = torch.empty(0)

    # We patch parent module imports to avoid hitting the hivemind PeerID issue
    with mock.patch.dict('sys.modules', {
        'hivemind': mock_hivemind,
        'transformers': mock_transformers,
        'petals.utils.misc': mock_misc,
        'petals': mock.MagicMock(__path__=["src/petals"], __spec__=None),
        'petals.client': mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    }):
        import petals.client.ptune
        with mock.patch.object(petals.client.ptune, '_original_register_parameter', nn.Module.register_parameter):
            from petals.client.ptune import PTuneMixin

            class DummyConfig:
                def __init__(self):
                    self.tuning_mode = "deep_ptune"
                    self.pre_seq_len = 5
                    self.hidden_size = 16
                    self.num_hidden_layers = 10

            class DummyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, 16)
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)

            assert model.intermediate_prompt_embeddings.embedding_dim == (10 - 1) * 16

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (batch_size, 5, 16)
            assert intermediate_prompts.shape == (10 - 1, batch_size, 5, 16)
