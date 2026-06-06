import sys
import os
sys.path.insert(0, os.path.abspath('src'))

# Disable version checks in petals/__init__.py
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

def test_ptune_mixin_num_hidden_layers():
    hivemind_mock = mock.MagicMock(__path__=["hivemind"], __spec__=None)
    hivemind_utils_mock = mock.MagicMock(__path__=["hivemind.utils"], __spec__=None)

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'hivemind.dht': mock.MagicMock(),
        'hivemind.moe': mock.MagicMock(),
        'hivemind.moe.client': mock.MagicMock(),
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
        'hivemind.moe.expert_uid': mock.MagicMock(),
        'hivemind.p2p': mock.MagicMock(),
        'hivemind.p2p.p2p_daemon': mock.MagicMock(),
        'hivemind.utils': hivemind_utils_mock,
        'hivemind.utils.logging': mock.MagicMock(),
        'hivemind.utils.tensor_deserializer': mock.MagicMock(),
        'petals.client.inference_session': mock.MagicMock(),
        'petals.client.remote_sequential': mock.MagicMock(),
        'petals.client.routing': mock.MagicMock(),
    }):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin

            class MyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=3
            )

            model = MyModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (2, 2, 5, 16), f"Wrong shape: {intermediate_prompts.shape}"

if __name__ == "__main__":
    test_ptune_mixin_num_hidden_layers()
