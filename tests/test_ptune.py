def test_ptune_intermediate_prompt_shape():
    import sys
    import types
    from unittest import mock
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    # Mock missing dependencies
    petals_mock = types.ModuleType("petals")
    petals_mock.__path__ = []
    petals_mock.__spec__ = None

    petals_client_mock = types.ModuleType("petals.client")
    petals_client_mock.__path__ = []
    petals_client_mock.__spec__ = None

    petals_utils_mock = types.ModuleType("petals.utils")
    petals_utils_mock.__path__ = []
    petals_utils_mock.__spec__ = None

    petals_utils_misc_mock = types.ModuleType("petals.utils.misc")
    petals_utils_misc_mock.__path__ = []
    petals_utils_misc_mock.__spec__ = None
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    hivemind_mock = types.ModuleType("hivemind")
    hivemind_mock.__path__ = []
    hivemind_mock.__spec__ = None
    hivemind_mock.get_logger = lambda name: None

    mocks = {
        "petals": petals_mock,
        "petals.client": petals_client_mock,
        "petals.utils": petals_utils_mock,
        "petals.utils.misc": petals_utils_misc_mock,
        "hivemind": hivemind_mock,
    }

    with mock.patch.dict("sys.modules", mocks):
        import petals.client.ptune as ptune_module

        # Mock the locally captured register_parameter
        with mock.patch.object(ptune_module, '_original_register_parameter', nn.Module.register_parameter):
            class DummyModel(nn.Module, ptune_module.PTuneMixin):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.word_embeddings = nn.Embedding(10, config.hidden_size)
                    self.init_prompts(config)

            config = PretrainedConfig(
                tuning_mode="deep_ptune",
                pre_seq_len=5,
                hidden_size=16,
                num_hidden_layers=4,
            )

            model = DummyModel(config)

            assert model.intermediate_prompt_embeddings.embedding_dim == (config.num_hidden_layers - 1) * config.hidden_size

            prompts, intermediate_prompts = model.get_prompt(batch_size=2)

            assert intermediate_prompts.shape == (config.num_hidden_layers, 2, config.pre_seq_len, config.hidden_size)

if __name__ == "__main__":
    test_ptune_intermediate_prompt_shape()
    print("Test passed.")
