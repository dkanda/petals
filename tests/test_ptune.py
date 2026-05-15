import sys
from unittest import mock
import torch
from transformers import PretrainedConfig

def test_ptune_intermediate_prompt_embeddings_shape():
    hivemind_mock = mock.MagicMock(__path__=[], __spec__=None)
    hivemind_moe_mock = mock.MagicMock(__path__=[], __spec__=None)
    hivemind_moe_client_mock = mock.MagicMock(__path__=[], __spec__=None)
    hivemind_mock.moe = hivemind_moe_mock
    hivemind_moe_mock.client = hivemind_moe_client_mock

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_utils_mock = mock.MagicMock(__path__=["src/petals/utils"], __spec__=None)
    petals_utils_misc_mock = mock.MagicMock()

    petals_mock.client = petals_client_mock
    petals_mock.utils = petals_utils_mock
    petals_utils_mock.misc = petals_utils_misc_mock

    mocks = {
        'hivemind': hivemind_mock,
        'hivemind.moe': hivemind_moe_mock,
        'hivemind.moe.client': hivemind_moe_client_mock,
        'hivemind.moe.client.remote_expert_worker': mock.MagicMock(),
        'hivemind.p2p': mock.MagicMock(),
        'hivemind.utils': mock.MagicMock(),
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'petals.utils': petals_utils_mock,
        'petals.utils.misc': petals_utils_misc_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        sys.modules['petals.utils.misc'].DUMMY = torch.zeros(1)

        from petals.client.ptune import PTuneConfig, PTuneMixin
        import petals.client.ptune as ptune_mod

        with mock.patch.object(ptune_mod, '_original_register_parameter', torch.nn.Module.register_parameter):
            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=8,
                hidden_size=16,
                num_hidden_layers=4,
            )

            class DummyModel(torch.nn.Module, PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = torch.nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            model = DummyModel(config)

            # Test shape reduction from num_hidden_layers to num_hidden_layers - 1
            assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([8, 48]), f"Expected shape [8, 48], got {model.intermediate_prompt_embeddings.weight.shape}"

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)
            assert prompts.shape == torch.Size([2, 8, 16]), f"Expected shape [2, 8, 16], got {prompts.shape}"
            assert intermediate_prompts.shape == torch.Size([4, 2, 8, 16]), f"Expected shape [4, 2, 8, 16], got {intermediate_prompts.shape}"
