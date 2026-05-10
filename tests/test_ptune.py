import sys
import types
from unittest import mock

def test_ptune_shapes():
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    # Mock dependencies before loading
    mock_hivemind = types.ModuleType('hivemind')
    mock_hivemind.__path__ = []
    mock_hivemind.get_logger = mock.MagicMock()

    mock_petals_utils_misc = types.ModuleType('petals.utils.misc')
    mock_petals_utils_misc.__path__ = []
    mock_petals_utils_misc.DUMMY = None

    with mock.patch.dict('sys.modules', {
        'hivemind': mock_hivemind,
        'petals.utils.misc': mock_petals_utils_misc,
    }):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class MockModel(nn.Module, ptune.PTuneMixin):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = nn.Embedding(10, config.hidden_size)
                with mock.patch.object(ptune, '_original_register_parameter', nn.Module.register_parameter):
                    self.init_prompts(config)

        config = PretrainedConfig(
            hidden_size=16,
            num_hidden_layers=4,
            tuning_mode="deep_ptune",
            pre_seq_len=5
        )

        model = MockModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (4, 2, 5, 16)
