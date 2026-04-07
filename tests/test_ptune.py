import torch
import pytest
import sys
from unittest import mock
from transformers import PretrainedConfig

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def get_mocks():
    return {
        'hivemind': MockPackage(),
        'hivemind.utils': MockPackage(),
        'hivemind.utils.logging': MockPackage(),
        'hivemind.moe': MockPackage(),
        'hivemind.moe.client': MockPackage(),
        'hivemind.moe.client.remote_expert_worker': MockPackage(),
        'hivemind.moe.expert_uid': MockPackage(),
        'hivemind.dht': MockPackage(),
        'hivemind.p2p': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings.datastructures': MockPackage(),
        'hivemind.utils.streaming': MockPackage(),
        'hivemind.utils.mpfuture': MockPackage(),
        'tensor_parallel': MockPackage(),
        'hivemind.proto': MockPackage(),
        'hivemind.utils.tensor_descr': MockPackage(),
        'hivemind.utils.asyncio': MockPackage(),
        'hivemind.moe.server.module_backend': MockPackage(),
        'hivemind.dht.routing': MockPackage(),
        'hivemind.p2p.p2p_daemon': MockPackage(),
        'tensor_parallel.slicing_configs': MockPackage(),
        'hivemind.utils.nested': MockPackage(),
        'hivemind.dht.node': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings.control': MockPackage(),
        'hivemind.moe.server': MockPackage(),
        'hivemind.moe.server.connection_handler': MockPackage(),
        'bitsandbytes': MockPackage(),
        'bitsandbytes.optim': MockPackage(),
        'tensor_parallel.tensor_parallel': MockPackage(),
        'speedtest': MockPackage(),
        'hivemind.compression': MockPackage(),
        'hivemind.compression.serialization': MockPackage(),
        'hivemind.p2p.p2p_daemon_bindings.utils': MockPackage(),
    }

def setup_module():
    sys.modules.update(get_mocks())

def teardown_module():
    for key in get_mocks().keys():
        sys.modules.pop(key, None)

class DummyWordEmbeddings:
    def __init__(self):
        self.weight = torch.empty(0, device='cpu', dtype=torch.float32)

def test_ptune():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune_module = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune_module
    spec.loader.exec_module(ptune_module)

    PTuneMixin = ptune_module.PTuneMixin
    PTuneConfig = ptune_module.PTuneConfig

    spec2 = importlib.util.spec_from_file_location("misc", "src/petals/utils/misc.py")
    misc_module = importlib.util.module_from_spec(spec2)
    sys.modules["misc"] = misc_module
    spec2.loader.exec_module(misc_module)
    DUMMY = misc_module.DUMMY

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = DummyWordEmbeddings()
            self.init_prompts(config)

    config = PretrainedConfig(tuning_mode="ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=4)
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 8)
    assert "DUMMY.to()" in repr(intermediate_prompts) or intermediate_prompts is DUMMY or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.numel() == 0)

def test_deep_ptune():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune_module = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune_module
    spec.loader.exec_module(ptune_module)

    PTuneMixin = ptune_module.PTuneMixin
    PTuneConfig = ptune_module.PTuneConfig

    spec2 = importlib.util.spec_from_file_location("misc", "src/petals/utils/misc.py")
    misc_module = importlib.util.module_from_spec(spec2)
    sys.modules["misc"] = misc_module
    spec2.loader.exec_module(misc_module)
    DUMMY = misc_module.DUMMY

    class DummyModel(PTuneMixin):
        def __init__(self, config):
            self.config = config
            self.word_embeddings = DummyWordEmbeddings()
            self.init_prompts(config)

    config = PretrainedConfig(tuning_mode="deep_ptune", pre_seq_len=5, hidden_size=8, num_hidden_layers=4)
    model = DummyModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 8)
    assert intermediate_prompts.shape == (4, 2, 5, 8)
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
