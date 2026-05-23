import sys
import os
from unittest import mock
import torch
import torch.nn as nn

# Clean up module imports for testing

def test_ptune_intermediate_prompts_shape():
    with mock.patch.dict('sys.modules', {
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }), mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
        # Mocking hivemind objects that aren't top-level exports
        import hivemind
        hivemind.PeerID = hivemind.p2p.PeerID if hasattr(hivemind, 'p2p') else mock.MagicMock()
        hivemind.MSGPackSerializer = hivemind.utils.MSGPackSerializer if hasattr(hivemind, 'utils') else mock.MagicMock()
        hivemind.get_logger = mock.MagicMock()

        # By placing this inside the test, and mocking heavy downstream dependencies
        # as suggested in memory, we can use the standard import.
        from petals.client.ptune import PTuneMixin

        class MockConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 4

        class MockModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        model = MockModel(MockConfig())
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])
