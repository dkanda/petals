import os
import sys

sys.path.insert(0, os.path.abspath('src'))

import unittest.mock as mock

mock_petals_utils_misc = mock.MagicMock()
mock_petals_utils_misc.DUMMY = "DUMMY_TENSOR"

# We mock `petals` at the root so it doesn't try to import everything in __init__.py
with mock.patch.dict('sys.modules', {
    'petals.utils.misc': mock_petals_utils_misc,
    'hivemind': mock.MagicMock(),
}):
    # load ptune module directly using importlib to avoid __init__ issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("petals.client.ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["petals.client.ptune"] = ptune
    spec.loader.exec_module(ptune)

    from petals.client.ptune import PTuneMixin

    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    class DummyConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 64
            self.num_hidden_layers = 10
            self.tuning_mode = "deep_ptune"
            self.pre_seq_len = 5

    class MockModel(nn.Module, PTuneMixin):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.word_embeddings = nn.Embedding(100, config.hidden_size)
            self.init_prompts(config)

def test_deep_ptune_intermediate_prompts_shape():
    """
    Test that the intermediate_prompts tensor returned by get_prompt
    has the correct shape using num_hidden_layers - 1.
    """
    config = DummyConfig()
    model = MockModel(config)
    prompts, intermediate_prompts = model.get_prompt(batch_size=2)

    # prompts shape should be (batch_size, pre_seq_len, hidden_size)
    assert prompts.shape == (2, 5, 64)

    # intermediate_prompts shape should be (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    assert intermediate_prompts.shape == (9, 2, 5, 64)
