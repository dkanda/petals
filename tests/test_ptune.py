import sys
import os
import torch
import torch.nn as nn
from unittest import mock

sys.path.insert(0, os.path.abspath('src'))

def test_ptune():
    mocks = {}
    for mod in [
        'hivemind',
        'hivemind.moe',
        'hivemind.moe.client',
        'hivemind.moe.client.remote_expert_worker',
        'hivemind.moe.server',
        'hivemind.moe.server.module_backend',
        'hivemind.moe.server.connection_handler',
        'hivemind.moe.expert_uid',
        'hivemind.p2p',
        'hivemind.p2p.p2p_daemon_bindings',
        'hivemind.p2p.p2p_daemon_bindings.control',
        'hivemind.p2p.p2p_daemon',
        'hivemind.dht',
        'hivemind.dht.node',
        'hivemind.proto',
        'hivemind.proto.runtime_pb2',
        'hivemind.utils',
        'hivemind.utils.asyncio',
        'hivemind.utils.logging',
        'hivemind.utils.mpfuture',
        'hivemind.utils.streaming',
        'hivemind.utils.nested',
        'hivemind.utils.tensor_descr',
        'hivemind.compression',
        'hivemind.compression.serialization',
        'tensor_parallel',
        'tensor_parallel.slicing_configs',
        'tensor_parallel.tensor_parallel',
    ]:
        mocks[mod] = mock.MagicMock()

    mocks['hivemind.p2p.p2p_daemon_bindings.control'].DEFAULT_MAX_MSG_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon_bindings.control'].MAX_UNARY_PAYLOAD_SIZE = 1000000
    mocks['hivemind.p2p.p2p_daemon'].DEFAULT_MAX_MSG_SIZE = 1000000

    with mock.patch.dict('sys.modules', mocks):
        from transformers import PretrainedConfig
        from petals.client.ptune import PTuneMixin

        class MockModel(nn.Module, PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            hidden_size=16,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=5
        )

        model = MockModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == torch.Size([2, 5, 16])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16])

if __name__ == '__main__':
    test_ptune()
    print('Test passed')
