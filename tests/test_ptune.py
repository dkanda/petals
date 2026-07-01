import torch
import torch.nn as nn
from petals.client.ptune import PTuneMixin

class DummyConfig:
    def __init__(self):
        self.tuning_mode = "deep_ptune"
        self.pre_seq_len = 10
        self.hidden_size = 16
        self.num_hidden_layers = 5

class MockModel(PTuneMixin):
    def __init__(self):
        self.config = DummyConfig()
        self.init_prompts(self.config)
        self.word_embeddings = type("Mock", (), {"weight": torch.zeros(1)})()

def test_ptune_intermediate_prompt_embeddings_shape():
    model = MockModel()

    # We expect pre_seq_len (10) for vocabulary, and embedding size of (num_hidden_layers - 1) * hidden_size = (5 - 1) * 16 = 64
    assert model.intermediate_prompt_embeddings.weight.shape == (10, 64)

    # get_prompt should return (prompts, intermediate_prompts)
    _, intermediate_prompts = model.get_prompt(batch_size=2)
    # the expected shape is (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size) = (4, 2, 10, 16)
    assert intermediate_prompts.shape == (4, 2, 10, 16)
