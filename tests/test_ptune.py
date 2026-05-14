import pytest
from unittest import mock

def test_ptune_intermediate_prompts_shape():
    import torch
    import torch.nn as nn
    import types

    petals_mock = types.ModuleType("petals")
    petals_mock.__path__ = ["src/petals"]
    petals_mock.__spec__ = None

    petals_client_mock = types.ModuleType("petals.client")
    petals_client_mock.__path__ = ["src/petals/client"]
    petals_client_mock.__spec__ = None
    petals_mock.client = petals_client_mock

    hivemind_mock = types.ModuleType("hivemind")
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    hivemind_mock.get_logger = mock.MagicMock()
    hivemind_mock.PeerID = mock.MagicMock()

    hivemind_moe_mock = types.ModuleType("hivemind.moe")
    hivemind_moe_mock.__path__ = []
    hivemind_moe_mock.__spec__ = None

    hivemind_moe_expert_uid_mock = types.ModuleType("hivemind.moe.expert_uid")
    hivemind_moe_expert_uid_mock.__path__ = []
    hivemind_moe_expert_uid_mock.__spec__ = None
    hivemind_moe_expert_uid_mock.ExpertUID = mock.MagicMock()

    mocks = {
        'petals': petals_mock,
        'petals.client': petals_client_mock,
        'hivemind': hivemind_mock,
        'hivemind.dht': mock.MagicMock(),
        'hivemind.p2p': mock.MagicMock(),
        'hivemind.utils': mock.MagicMock(),
        'hivemind.moe': hivemind_moe_mock,
        'hivemind.moe.expert_uid': hivemind_moe_expert_uid_mock,
    }

    with mock.patch.dict('sys.modules', mocks):
        from petals.client.ptune import PTuneMixin
        import petals.client.ptune
        from transformers import PretrainedConfig

        with mock.patch.object(petals.client.ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            class DummyConfig:
                tuning_mode = "deep_ptune"
                pre_seq_len = 5
                hidden_size = 16
                num_hidden_layers = 4

            class DummyModel(PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)

            config = DummyConfig()
            model = DummyModel(config)

            model.init_prompts(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (2, 5, 16)
            assert intermediate_prompts.shape == (4, 2, 5, 16)
            assert torch.allclose(intermediate_prompts[0], prompts)
