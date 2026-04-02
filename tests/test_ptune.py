import unittest.mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys

# We need to mock hivemind because it's imported in ptune
class MockPackage:
    __path__ = []
    __spec__ = None
    __name__ = 'hivemind'
    PeerID = str
    def __getattr__(self, name):
        return unittest.mock.MagicMock()

mock_modules = {
    'hivemind': MockPackage(),
    'hivemind.moe': MockPackage(),
    'hivemind.moe.client': MockPackage(),
    'hivemind.moe.client.remote_expert_worker': MockPackage(),
    'hivemind.moe.expert_uid': MockPackage(),
    'hivemind.moe.server': MockPackage(),
    'hivemind.moe.server.connection_handler': MockPackage(),
    'hivemind.moe.server.module_backend': MockPackage(),
    'hivemind.p2p': MockPackage(),
    'hivemind.p2p.p2p_daemon': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
    'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
    'hivemind.utils': MockPackage(),
    'hivemind.utils.nested': MockPackage(),
    'hivemind.utils.tensor_descr': MockPackage(),
    'hivemind.utils.logging': MockPackage(),
    'hivemind.utils.asyncio': MockPackage(),
    'hivemind.utils.streaming': MockPackage(),
    'hivemind.utils.mpfuture': MockPackage(),
    'hivemind.dht': MockPackage(),
    'hivemind.dht.routing': MockPackage(),
    'hivemind.dht.node': MockPackage(),
    'hivemind.proto': MockPackage(),
    'tensor_parallel': MockPackage(),
    'tensor_parallel.slicing_configs': MockPackage(),
    'tensor_parallel.tensor_parallel': MockPackage(),
    'hivemind.compression': MockPackage(),
    'hivemind.compression.serialization': MockPackage(),
}

def setup_module():
    unittest.mock.patch.dict('sys.modules', mock_modules).start()

def teardown_module():
    unittest.mock.patch.stopall()

# import it here so that the mock takes effect
def get_ptune_mixin():
    from petals.client.ptune import PTuneMixin
    return PTuneMixin

class TestPTune(unittest.TestCase):

    def test_deep_ptune_embeddings(self):
        PTuneMixin = get_ptune_mixin()
        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=8,
            num_hidden_layers=4,
        )

        model = DummyModel(config)

        # Check that intermediate prompt embeddings has correct shape:
        # (num_hidden_layers - 1) * hidden_size
        expected_embedding_dim = (4 - 1) * 8
        self.assertEqual(model.intermediate_prompt_embeddings.embedding_dim, expected_embedding_dim)

        # get_prompt with batch size 2
        prompts, intermediate_prompts = model.get_prompt(2)

        # prompts should be [2, pre_seq_len, hidden_size]
        self.assertEqual(prompts.shape, (2, 5, 8))

        # intermediate_prompts should be [num_hidden_layers, batch_size, pre_seq_len, hidden_size]
        # (This is due to the `.permute([2, 0, 1, 3])` happening inside get_prompt)
        self.assertEqual(intermediate_prompts.shape, (4, 2, 5, 8))

        # first layer should be all zeros
        self.assertTrue(torch.all(intermediate_prompts[0, :, :, :] == 0))

        # other layers should not be entirely zeros (assuming random init didn't give exact zeros)
        self.assertFalse(torch.all(intermediate_prompts[1:, :, :, :] == 0))

if __name__ == '__main__':
    unittest.main()
