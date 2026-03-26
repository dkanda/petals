import sys
import unittest
from unittest.mock import MagicMock

class MockPackage(MagicMock):
    __path__ = []
    __spec__ = None

sys.modules['hivemind'] = MockPackage()
sys.modules['hivemind.dht'] = MockPackage()
sys.modules['hivemind.dht.node'] = MockPackage()
sys.modules['hivemind.moe'] = MockPackage()
sys.modules['hivemind.moe.client'] = MockPackage()
sys.modules['hivemind.moe.client.remote_expert_worker'] = MockPackage()
sys.modules['hivemind.moe.server'] = MockPackage()
sys.modules['hivemind.moe.server.connection_handler'] = MockPackage()
sys.modules['hivemind.moe.server.module_backend'] = MockPackage()
sys.modules['hivemind.moe.expert_uid'] = MockPackage()
sys.modules['hivemind.p2p'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon_bindings.control'] = MockPackage()
sys.modules['hivemind.p2p.p2p_daemon'] = MockPackage()
sys.modules['hivemind.proto'] = MockPackage()
sys.modules['hivemind.utils'] = MockPackage()
sys.modules['hivemind.utils.asyncio'] = MockPackage()
sys.modules['hivemind.utils.logging'] = MockPackage()
sys.modules['hivemind.utils.nested'] = MockPackage()
sys.modules['hivemind.utils.streaming'] = MockPackage()
sys.modules['hivemind.utils.tensor_descr'] = MockPackage()
sys.modules['hivemind.utils.mpfuture'] = MockPackage()
sys.modules['hivemind.compression'] = MockPackage()
sys.modules['hivemind.compression.serialization'] = MockPackage()
sys.modules['transformers'] = MockPackage()
sys.modules['transformers.modeling_attn_mask_utils'] = MockPackage()
sys.modules['transformers.models'] = MockPackage()
sys.modules['transformers.models.bloom'] = MockPackage()
sys.modules['transformers.models.bloom.modeling_bloom'] = MockPackage()
sys.modules['transformers.models.llama'] = MockPackage()
sys.modules['transformers.models.llama.modeling_llama'] = MockPackage()
sys.modules['transformers.models.mixtral'] = MockPackage()
sys.modules['transformers.models.mixtral.modeling_mixtral'] = MockPackage()
sys.modules['transformers.models.falcon'] = MockPackage()
sys.modules['transformers.models.falcon.modeling_falcon'] = MockPackage()
sys.modules['transformers.models.deepseek'] = MockPackage()
sys.modules['transformers.models.deepseek.modeling_deepseek'] = MockPackage()
sys.modules['transformers.utils'] = MockPackage()
sys.modules['dijkstar'] = MockPackage()
sys.modules['bitsandbytes'] = MockPackage()
sys.modules['speedtest'] = MockPackage()
sys.modules['numpy'] = MockPackage()
sys.modules['pydantic'] = MockPackage()
sys.modules['pydantic.v1'] = MockPackage()
sys.modules['fastapi'] = MockPackage()
sys.modules['async_timeout'] = MockPackage()
sys.modules['tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.tensor_parallel'] = MockPackage()
sys.modules['tensor_parallel.slicing_configs'] = MockPackage()
sys.modules['accelerate'] = MockPackage()

# We don't want to actually import petals and all its dependencies, just ptune.py
# However, ptune.py requires petals.utils.misc.DUMMY and transformers.PretrainedConfig
# We'll mock the whole petals module and just load the source of ptune.py using importlib
import importlib.util
import sys

import torch
import torch.nn as nn

petals_mock = MockPackage()
sys.modules['petals'] = petals_mock
sys.modules['petals.utils'] = MockPackage()
sys.modules['petals.utils.misc'] = MockPackage()

import petals.utils.misc
petals.utils.misc.DUMMY = torch.empty(0)

# Also mock PretrainedConfig as a simple class
class PretrainedConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.tuning_mode = kwargs.get('tuning_mode', None)
        self.pre_seq_len = kwargs.get('pre_seq_len', 0)

transformers_mock = MockPackage()
transformers_mock.PretrainedConfig = PretrainedConfig
sys.modules['transformers'] = transformers_mock

# Now we can manually load ptune.py
spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
ptune = importlib.util.module_from_spec(spec)
sys.modules["petals.client.ptune"] = ptune
spec.loader.exec_module(ptune)

PTuneMixin = ptune.PTuneMixin

class DummyModel(nn.Module, PTuneMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = MagicMock()
        self.word_embeddings.weight = MagicMock()
        self.word_embeddings.weight.device = "cpu"
        self.word_embeddings.weight.dtype = torch.float32

class TestPTuneMixin(unittest.TestCase):
    def test_ptune(self):
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="ptune"
        )
        model = DummyModel(config)
        model.init_prompts(config)

        self.assertTrue(hasattr(model, 'prompt_embeddings'))
        self.assertFalse(hasattr(model, 'intermediate_prompt_embeddings'))

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
        # DUMMY tensor from petals.utils.misc has 0 size
        self.assertEqual(
            intermediate_prompts.numel() if isinstance(intermediate_prompts, torch.Tensor) else 0,
            0
        )

    def test_deep_ptune(self):
        config = PretrainedConfig(
            hidden_size=64,
            num_hidden_layers=4,
            pre_seq_len=8,
            tuning_mode="deep_ptune"
        )
        model = DummyModel(config)
        model.init_prompts(config)

        self.assertTrue(hasattr(model, 'prompt_embeddings'))
        self.assertTrue(hasattr(model, 'intermediate_prompt_embeddings'))

        # Check shape of intermediate_prompt_embeddings.weight
        self.assertEqual(
            model.intermediate_prompt_embeddings.weight.shape,
            (config.pre_seq_len, (config.num_hidden_layers - 1) * config.hidden_size)
        )

        batch_size = 2
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        self.assertEqual(prompts.shape, (batch_size, config.pre_seq_len, config.hidden_size))
        self.assertEqual(
            intermediate_prompts.shape,
            (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
        )

        # Ensure the first layer prompts are all zero padding
        self.assertTrue(torch.all(intermediate_prompts[0] == 0))

        # Ensure the rest of the layers are non-zero (since they come from initialized embeddings)
        self.assertFalse(torch.all(intermediate_prompts[1:] == 0))

if __name__ == '__main__':
    unittest.main()
