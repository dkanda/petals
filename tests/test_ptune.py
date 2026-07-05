import sys
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# According to memory: "When unit testing modules with uninstallable dependencies (like hivemind on Python 3.12)..."
# "wrap the import in a try... except ImportError: pytest.skip('...', allow_module_level=True) block."
# But the skip needs to happen BEFORE the global import of petals.client.ptune which triggers the failure.

try:
    # Just try to import PeerID which fails locally to skip
    from hivemind import PeerID
    from petals.client.ptune import PTuneMixin
except ImportError:
    pytest.skip("hivemind dependencies missing locally, skipping test", allow_module_level=True)

class MockModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(100, config.hidden_size)
        self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    config = PretrainedConfig(
        hidden_size=16,
        num_hidden_layers=4,
        pre_seq_len=8,
        tuning_mode="deep_ptune"
    )

    model = MockModel(config)

    # Assert intermediate prompt embedding shape
    expected_embedding_dim = (config.num_hidden_layers - 1) * config.hidden_size
    assert model.intermediate_prompt_embeddings.embedding_dim == expected_embedding_dim

    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # Layer 0 prefix: (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == torch.Size([2, 8, 16])

    # Layer 1 to N prefix: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    # So for 4 layers: (3, 2, 8, 16)
    assert intermediate_prompts.shape == torch.Size([3, 2, 8, 16])
