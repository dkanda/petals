import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
import sys
import unittest.mock as mock

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def setup_module():
    sys.modules["hivemind"] = MockPackage()
    sys.modules["hivemind.get_logger"] = mock.MagicMock()
    sys.modules["hivemind.moe"] = MockPackage()
    sys.modules["hivemind.moe.client"] = MockPackage()
    sys.modules["hivemind.moe.client.remote_expert_worker"] = MockPackage()
    sys.modules["hivemind.moe.expert_uid"] = MockPackage()
    sys.modules["hivemind.moe.server"] = MockPackage()
    sys.modules["hivemind.moe.server.module_backend"] = MockPackage()
    sys.modules["hivemind.moe.server.connection_handler"] = MockPackage()
    sys.modules["hivemind.p2p"] = MockPackage()
    sys.modules["hivemind.p2p.p2p_daemon"] = MockPackage()
    sys.modules["hivemind.p2p.p2p_daemon_bindings"] = MockPackage()
    sys.modules["hivemind.p2p.p2p_daemon_bindings.datastructures"] = MockPackage()
    sys.modules["hivemind.utils"] = MockPackage()
    sys.modules["hivemind.utils.mpfuture"] = MockPackage()
    sys.modules["hivemind.utils.streaming"] = MockPackage()
    sys.modules["hivemind.utils.nested"] = MockPackage()
    sys.modules["hivemind.utils.asyncio"] = MockPackage()
    sys.modules["hivemind.utils.tensor_descr"] = MockPackage()
    sys.modules["hivemind.utils.logging"] = MockPackage()
    sys.modules["hivemind.dht"] = MockPackage()
    sys.modules["hivemind.dht.routing"] = MockPackage()
    sys.modules["hivemind.dht.node"] = MockPackage()
    sys.modules["hivemind.compression"] = MockPackage()
    sys.modules["hivemind.compression.base"] = MockPackage()
    sys.modules["hivemind.proto"] = MockPackage()
    sys.modules["hivemind.proto.runtime_pb2"] = MockPackage()
    sys.modules["tensor_parallel"] = MockPackage()
    sys.modules["tensor_parallel.slicing_configs"] = MockPackage()
    sys.modules["tensor_parallel.tensor_parallel"] = MockPackage()
    sys.modules["petals"] = MockPackage()

def teardown_module():
    keys = list(sys.modules.keys())
    for k in keys:
        if k.startswith("hivemind") or k.startswith("tensor_parallel") or k.startswith("petals"):
            del sys.modules[k]

def test_deep_ptune():
    with mock.patch.dict("sys.modules", {"petals.utils": MockPackage(), "petals.utils.misc": MockPackage()}):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        ptune.get_logger = mock.MagicMock()

        # Inject realistic DUMMY tensor so `.to(dtype)` works properly instead of generating a new MockPackage
        ptune.DUMMY = torch.zeros(0)
        sys.modules["petals.utils.misc"].DUMMY = ptune.DUMMY

        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=3,
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (3, 2, 5, 16)
        assert torch.all(intermediate_prompts[0] == 0)

def test_ptune():
    with mock.patch.dict("sys.modules", {"petals.utils": MockPackage(), "petals.utils.misc": MockPackage()}):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        ptune.get_logger = mock.MagicMock()

        # Inject realistic DUMMY tensor
        ptune.DUMMY = torch.zeros(0)
        sys.modules["petals.utils.misc"].DUMMY = ptune.DUMMY

        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode="ptune",
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=3,
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts is ptune.DUMMY or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0)
