import sys
import types
from unittest import mock

hivemind_mock = types.ModuleType('hivemind')
hivemind_mock.__path__ = []
hivemind_mock.__spec__ = None
hivemind_mock.get_logger = mock.MagicMock()
hivemind_mock.PeerID = mock.MagicMock()

petals_mock = types.ModuleType("petals")
petals_mock.__path__ = ["src/petals"]
petals_mock.__spec__ = None

petals_utils_mock = types.ModuleType("petals.utils")
petals_utils_mock.__path__ = ["src/petals/utils"]
petals_utils_mock.__spec__ = None
petals_mock.utils = petals_utils_mock

petals_utils_misc_mock = types.ModuleType("petals.utils.misc")
petals_utils_misc_mock.DUMMY = "DUMMY_VALUE"
petals_utils_mock.misc = petals_utils_misc_mock

petals_client_mock = types.ModuleType("petals.client")
petals_client_mock.__path__ = ["src/petals/client"]
petals_client_mock.__spec__ = None
petals_mock.client = petals_client_mock

sys.path.insert(0, "src")

def test_ptune_intermediate_prompt_shape():
    with mock.patch.dict(sys.modules, {
        'hivemind': hivemind_mock,
        'petals': petals_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
        'petals.client': petals_client_mock,
    }):
        from petals.client.ptune import PTuneMixin
        import petals.client.ptune as ptune_module
        import torch
        import torch.nn as nn
        from transformers import PretrainedConfig

        class DummyModel(PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                self.init_prompts(config)

        config = PretrainedConfig(
            tuning_mode='deep_ptune',
            pre_seq_len=5,
            hidden_size=16,
            num_hidden_layers=4,
        )

        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            model = DummyModel(config)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 16]), f"Expected (3, 2, 5, 16) but got {intermediate_prompts.shape}"
        assert prompts.shape == torch.Size([2, 5, 16]), f"Expected (2, 5, 16) but got {prompts.shape}"

if __name__ == "__main__":
    test_ptune_intermediate_prompt_shape()
