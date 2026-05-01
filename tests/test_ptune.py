import sys
import unittest.mock as mock
import importlib.util

mock_modules = {
    'hivemind': mock.MagicMock(),
    'transformers': mock.MagicMock(),
    'petals.utils.misc': mock.MagicMock(),
}

import torch
import torch.nn as nn

class MockMisc:
    DUMMY = torch.empty(0)
mock_modules['petals.utils.misc'] = MockMisc

with mock.patch.dict('sys.modules', mock_modules):
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)

    # We DO NOT mock nn.Module.register_parameter here to see if real torch allows it
    # Since we are using real torch.nn we need register_parameter to work normally for nn.Embedding
    original_register_parameter = nn.Module.register_parameter

    try:
        spec.loader.exec_module(ptune)
    finally:
        pass # keep it real

class DummyConfig:
    def __init__(self):
        self.pre_seq_len = 5
        self.tuning_mode = "deep_ptune"
        self.num_hidden_layers = 4
        self.hidden_size = 8

class MockModel(nn.Module, ptune.PTuneMixin):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.word_embeddings = nn.Embedding(10, 8)

        # Manually bypass init_empty_weights logic
        with ptune.force_non_empty_weights():
            self.init_prompts(self.config)

def test_deep_ptune_bugfix():
    m = MockModel()

    assert m.intermediate_prompt_embeddings.embedding_dim == (4 - 1) * 8

    prompts, intermediate_prompts = m.get_prompt(batch_size=2)
    assert prompts.shape == (2, 5, 8)
    assert intermediate_prompts.shape == (4, 2, 5, 8)
    assert torch.all(intermediate_prompts[0] == 0)

if __name__ == "__main__":
    test_deep_ptune_bugfix()
    print("Test passed!")
