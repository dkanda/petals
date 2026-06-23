import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from unittest import mock
import sys

# We need to mock hivemind because the latest published wheels fail on python 3.12
# with C++ compile errors during source installation, and the pre-built 1.1.12 wheel
# lacks 'PeerID' causing ImportError.
# We'll use mock.patch.dict within the test file to avoid polluting global scope.
def test_ptune_intermediate_prompts_shape():
    mock_hivemind = mock.MagicMock()
    mock_hivemind.p2p.p2p_daemon_bindings.control.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_hivemind.p2p.p2p_daemon.DEFAULT_MAX_MSG_SIZE = 1000000
    mock_tensor_parallel = mock.MagicMock()

    mocks = {
        'hivemind': mock_hivemind,
        'hivemind.p2p': mock_hivemind.p2p,
        'hivemind.p2p.p2p_daemon_bindings': mock_hivemind.p2p.p2p_daemon_bindings,
        'hivemind.p2p.p2p_daemon_bindings.control': mock_hivemind.p2p.p2p_daemon_bindings.control,
        'hivemind.p2p.p2p_daemon': mock_hivemind.p2p.p2p_daemon,
        'hivemind.utils': mock_hivemind.utils,
        'hivemind.utils.logging': mock_hivemind.utils.logging,
        'hivemind.utils.nested': mock_hivemind.utils.nested,
        'hivemind.moe': mock_hivemind.moe,
        'hivemind.moe.client': mock_hivemind.moe.client,
        'hivemind.moe.client.remote_expert_worker': mock_hivemind.moe.client.remote_expert_worker,
        'hivemind.moe.expert_uid': mock_hivemind.moe.expert_uid,
        'hivemind.dht': mock_hivemind.dht,
        'hivemind.dht.node': mock_hivemind.dht.node,
        'hivemind.proto': mock_hivemind.proto,
        'hivemind.proto.runtime_pb2': mock_hivemind.proto.runtime_pb2,
        'hivemind.utils.asyncio': mock_hivemind.utils.asyncio,
        'hivemind.utils.mpfuture': mock_hivemind.utils.mpfuture,
        'hivemind.utils.streaming': mock_hivemind.utils.streaming,
        'hivemind.utils.tensor_descr': mock_hivemind.utils.tensor_descr,
        'hivemind.compression': mock_hivemind.compression,
        'hivemind.compression.serialization': mock_hivemind.compression.serialization,
        'hivemind.moe.server': mock_hivemind.moe.server,
        'hivemind.moe.server.module_backend': mock_hivemind.moe.server.module_backend,
        'hivemind.moe.server.connection_handler': mock_hivemind.moe.server.connection_handler,
        'tensor_parallel': mock_tensor_parallel,
        'tensor_parallel.slicing_configs': mock_tensor_parallel.slicing_configs,
        'tensor_parallel.tensor_parallel': mock_tensor_parallel.tensor_parallel,
    }

    with mock.patch.dict('sys.modules', mocks):
        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            from petals.client.ptune import PTuneMixin
            import petals.utils.misc

            # Use DUMMY from petals
            petals.utils.misc.DUMMY = torch.empty(0)

            class DummyModel(PTuneMixin, nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=4,
            )

            model = DummyModel(config)

            # Check initialization shapes
            assert model.prompt_embeddings.weight.shape == (5, 16)
            assert model.intermediate_prompt_embeddings.weight.shape == (5, 3 * 16)

            # Check get_prompt shapes
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            # prompts shape: (batch_size, pre_seq_len, hidden_size)
            assert prompts.shape == (2, 5, 16)

            # intermediate_prompts shape: (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
            assert intermediate_prompts.shape == (3, 2, 5, 16)
