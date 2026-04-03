import unittest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

import sys
import unittest.mock as mock
import importlib.util

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def load_isolated_ptune():
    mock_modules = {
        'hivemind': MockPackage(),
        'hivemind.moe': MockPackage(),
        'hivemind.moe.client': MockPackage(),
        'hivemind.moe.client.remote_expert_worker': MockPackage(),
        'hivemind.utils': MockPackage(),
        'hivemind.utils.mpfuture': MockPackage(),
        'hivemind.moe.server': MockPackage(),
        'hivemind.moe.server.module_backend': MockPackage(),
        'hivemind.moe.expert_uid': MockPackage(),
        'hivemind.utils.nested': MockPackage(),
        'hivemind.dht': MockPackage(),
        'hivemind.dht.routing': MockPackage(),
        'hivemind.p2p': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings.datastructures': MockPackage(),
        'hivemind.p2p.p2p_daemon': MockPackage(),
        'hivemind.proto': MockPackage(),
        'hivemind.proto.runtime_pb2': MockPackage(),
        'hivemind.utils.tensor_deserialization': MockPackage(),
        'hivemind.utils.asyncio': MockPackage(),
        'hivemind.utils.tensor_descr': MockPackage(),
        'tensor_parallel': MockPackage(),
        'tensor_parallel.slicing_configs': MockPackage(),
        'tensor_parallel.tensor_parallel': MockPackage(),
        'petals': MockPackage(),
        'petals.utils': MockPackage(),
        'petals.utils.misc': MockPackage(),
    }

    with mock.patch.dict('sys.modules', mock_modules):
        # Inject dummy into module namespace to bypass import of petals.utils.misc.DUMMY
        dummy = mock.MagicMock()
        setattr(sys.modules['petals.utils.misc'], 'DUMMY', dummy)

        # Load isolated module
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune_module)

        return ptune_module.PTuneMixin, dummy

PTuneMixin, DUMMY = load_isolated_ptune()

class MockWordEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))

class MockModelConfig(PretrainedConfig):
    def __init__(self, hidden_size=64, num_hidden_layers=4, tuning_mode=None, pre_seq_len=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len

class MockModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = MockWordEmbeddings(1000, config.hidden_size)
        self.init_prompts(config)

class TestPTune(unittest.TestCase):
    def test_ptune_shapes(self):
        config = MockModelConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=4)
        model = MockModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Test basic ptune prompts
        self.assertEqual(prompts.shape, (batch_size, 5, 64))
        self.assertTrue("DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY)

    def test_deep_ptune_shapes(self):
        config = MockModelConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=64, num_hidden_layers=4)
        model = MockModel(config)

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # Test base prompt shape
        self.assertEqual(prompts.shape, (batch_size, 5, 64))

        # Test intermediate prompt shapes
        # Should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        self.assertEqual(intermediate_prompts.shape, (4, batch_size, 5, 64))

        # Check that the first layer (index 0) is all zeros (the padding)
        padding_layer = intermediate_prompts[0]
        self.assertTrue(torch.all(padding_layer == 0))

        # Check that subsequent layers are not all zeros (assuming random init)
        # We test layer 1, which corresponds to the first actual layer in intermediate_prompt_embeddings
        self.assertFalse(torch.all(intermediate_prompts[1] == 0))

if __name__ == '__main__':
    unittest.main()
