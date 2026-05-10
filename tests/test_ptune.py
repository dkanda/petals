def test_deep_ptune():
    import pytest
    import torch
    from unittest import mock
    import sys
    from transformers import PretrainedConfig
    import torch.nn as nn
    import importlib.util
    import types

    hivemind = types.ModuleType("hivemind")
    hivemind.__path__ = []
    hivemind.get_logger = mock.MagicMock()

    petals_utils_misc = types.ModuleType("petals.utils.misc")
    petals_utils_misc.DUMMY = mock.MagicMock()

    sys_modules = {
        'hivemind': hivemind,
        'petals.utils.misc': petals_utils_misc,
    }

    with mock.patch.dict('sys.modules', sys_modules):
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 5
                self.num_hidden_layers = 4
                self.hidden_size = 16

        with mock.patch.object(ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
            class DummyModel(ptune.PTuneMixin):
                def __init__(self, config):
                    self.config = config
                    self.word_embeddings = mock.MagicMock()
                    self.word_embeddings.weight = mock.MagicMock()
                    self.word_embeddings.weight.device = "cpu"
                    self.word_embeddings.weight.dtype = torch.float32
                    self.init_prompts(config)

            config = DummyConfig()
            model = DummyModel(config)
            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
            assert intermediate_prompts.shape == (config.num_hidden_layers, batch_size, config.pre_seq_len, config.hidden_size)
