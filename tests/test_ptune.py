import sys
import types
from unittest import mock

# Mock deeply nested dependencies
class PeerID: pass
class MockHivemind(types.ModuleType):
    PeerID = PeerID
    MSGPackSerializer = mock.Mock()
    anext = mock.Mock()
    deserialize_torch_tensor = mock.Mock()
    serialize_torch_tensor = mock.Mock()
    P2PContext = mock.Mock()
    @staticmethod
    def get_logger(name):
        return mock.Mock()

hm_mock = MockHivemind("hivemind")
hm_mock.__path__ = []

mock_petals = types.ModuleType("petals")
mock_petals.__path__ = []
mock_petals_utils = types.ModuleType("petals.utils")
mock_petals_utils.__path__ = []
mock_petals_utils_misc = types.ModuleType("petals.utils.misc")

# Perform mutations inside test to satisfy memory rules
def test_ptune_intermediate_shapes():
    import torch
    mock_petals_utils_misc.DUMMY = torch.empty(0)

    with mock.patch.dict("sys.modules", {
        "hivemind": hm_mock,
        "petals": mock_petals,
        "petals.utils": mock_petals_utils,
        "petals.utils.misc": mock_petals_utils_misc
    }):
        # Mock register_parameter
        from transformers import PretrainedConfig
        import importlib.util

        spec = importlib.util.spec_from_file_location("ptune_isolated", "src/petals/client/ptune.py")
        ptune_module = importlib.util.module_from_spec(spec)
        with mock.patch.dict("sys.modules", {"ptune_isolated": ptune_module}):
            spec.loader.exec_module(ptune_module)

            PTuneMixin = ptune_module.PTuneMixin

            class DummyConfig(PretrainedConfig):
                def __init__(self, tuning_mode=None, pre_seq_len=0, hidden_size=64, num_hidden_layers=3, **kwargs):
                    super().__init__(**kwargs)
                    self.tuning_mode = tuning_mode
                    self.pre_seq_len = pre_seq_len
                    self.hidden_size = hidden_size
                    self.num_hidden_layers = num_hidden_layers

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = mock.Mock()
                    self.word_embeddings.weight = mock.Mock()
                    self.word_embeddings.weight.device = "cpu"
                    self.word_embeddings.weight.dtype = torch.float32
                    self.init_prompts(config)

            config = DummyConfig(tuning_mode="deep_ptune", pre_seq_len=5, num_hidden_layers=3)
            with mock.patch.object(ptune_module, "_original_register_parameter", torch.nn.Module.register_parameter):
                model = DummyModel(config)

            # Assert init_prompts layer size
            assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([5, 128])

            # Assert get_prompt output size
            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == torch.Size([2, 5, 64])
            assert intermediate_prompts.shape == torch.Size([2, 2, 5, 64])
