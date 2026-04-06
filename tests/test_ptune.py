import sys
import unittest
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module():
    sys.modules['hivemind'] = MockPackage()

def teardown_module():
    sys.modules.pop('hivemind', None)

class TestPTune(unittest.TestCase):
    def test_deep_ptune_shapes(self):
        import torch
        from transformers import PretrainedConfig

        sys.modules['hivemind'] = MockPackage()

        # Explicitly mock necessary modules to avoid ModuleNotFoundError when importing petals
        sys.modules['hivemind.moe'] = MockPackage()
        sys.modules['hivemind.moe.client'] = MockPackage()
        sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
        sys.modules['hivemind.moe.expert_uid'] = MockPackage()
        sys.modules['hivemind.moe.server'] = MockPackage()
        sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
        sys.modules['hivemind.dht'] = MockPackage()
        sys.modules['hivemind.p2p'] = MockPackage()
        sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
        sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
        sys.modules['hivemind.p2p.p2p_daemon_bindings.datastructures'] = MockPackage()
        sys.modules['hivemind.utils'] = MockPackage()
        sys.modules['hivemind.utils.asyncio'] = MockPackage()
        sys.modules['hivemind.utils.mpfuture'] = MockPackage()
        sys.modules['hivemind.utils.nested'] = MockPackage()
        sys.modules['hivemind.utils.streaming'] = MockPackage()
        sys.modules['hivemind.proto'] = MockPackage()
        sys.modules['tensor_parallel'] = MockPackage()
        sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
        sys.modules['speedtest'] = MockPackage()

        # Isolate the target test module using standard techniques for testing submodules
        # where the parent package imports deep broken dependencies.
        # This replaces the builtins.__import__ hack.
        import importlib.util
        sys.modules['petals.utils.misc'] = type(sys)('misc')
        sys.modules['petals.utils.misc'].DUMMY = torch.empty(0)

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune_module
        spec.loader.exec_module(ptune_module)
        PTuneMixin = ptune_module.PTuneMixin

        class MockModel(PTuneMixin):
            def __init__(self, cfg):
                self.config = cfg
                self.init_prompts(cfg)
                self.word_embeddings = type('obj', (object,), {'weight': torch.empty(0, dtype=torch.float32, device='cpu')})

        config = PretrainedConfig()
        config.tuning_mode = "deep_ptune"
        config.pre_seq_len = 5
        config.hidden_size = 10
        config.num_hidden_layers = 4

        model = MockModel(config)

        # Check intermediate_prompt_embeddings shape
        self.assertEqual(
            model.intermediate_prompt_embeddings.weight.shape,
            (config.pre_seq_len, (config.num_hidden_layers - 1) * config.hidden_size)
        )

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        # Check get_prompt returns correctly shaped tensors
        self.assertEqual(
            intermediate_prompts.shape,
            (config.num_hidden_layers, 2, config.pre_seq_len, config.hidden_size)
        )

if __name__ == '__main__':
    unittest.main()
