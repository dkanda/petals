import sys
import os
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

mock_hivemind = mock.MagicMock()
# Explicitly map all hivemind submodules
hivemind_modules = [
    'hivemind',
    'hivemind.compression',
    'hivemind.compression.serialization',
    'hivemind.dht',
    'hivemind.dht.node',
    'hivemind.moe',
    'hivemind.moe.client',
    'hivemind.moe.client.remote_expert_worker',
    'hivemind.moe.expert_uid',
    'hivemind.moe.server',
    'hivemind.moe.server.connection_handler',
    'hivemind.moe.server.module_backend',
    'hivemind.p2p',
    'hivemind.p2p.p2p_daemon',
    'hivemind.p2p.p2p_daemon_bindings',
    'hivemind.p2p.p2p_daemon_bindings.control',
    'hivemind.proto',
    'hivemind.proto.runtime_pb2',
    'hivemind.utils',
    'hivemind.utils.asyncio',
    'hivemind.utils.mpfuture',
    'hivemind.utils.nested',
    'hivemind.utils.streaming',
    'hivemind.utils.tensor_descr',
    'hivemind.utils.logging'
]

for m in hivemind_modules:
    sys.modules[m] = mock_hivemind

mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
mock_hivemind.utils.logging.get_logger = lambda *args, **kwargs: mock.MagicMock()

sys.modules['tensor_parallel'] = mock.MagicMock()
sys.modules['tensor_parallel.slicing_configs'] = mock.MagicMock()
sys.modules['tensor_parallel.tensor_parallel'] = mock.MagicMock()

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

import transformers.utils.import_utils
transformers.utils.import_utils.is_torch_fx_available = lambda: False

from transformers import PretrainedConfig
from petals.client.ptune import PTuneMixin

class DummyWordEmbeddings:
    def __init__(self):
        self.weight = torch.empty(0, dtype=torch.float32, device="cpu")

class DummyModel(PTuneMixin):
    def __init__(self):
        self.config = PretrainedConfig(
            hidden_size=16,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=8
        )
        self.word_embeddings = DummyWordEmbeddings()
        self.init_prompts(self.config)

def test_ptune_intermediate_prompts_shape():
    model = DummyModel()
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

    # After our fix, the shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    # i.e., (4 - 1, 2, 8, 16) -> (3, 2, 8, 16)
    expected_shape = (model.config.num_hidden_layers - 1, batch_size, model.config.pre_seq_len, model.config.hidden_size)
    assert intermediate_prompts.shape == expected_shape, f"Expected shape {expected_shape}, got {intermediate_prompts.shape}"

if __name__ == "__main__":
    test_ptune_intermediate_prompts_shape()
    print("Test passed!")
