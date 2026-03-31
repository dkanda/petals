import unittest
import torch
import torch.nn as nn

from petals.client.ptune import PTuneMixin
from petals.utils.misc import DUMMY

class DummyConfig:
    def __init__(self, tuning_mode, pre_seq_len, hidden_size, num_hidden_layers):
        self.tuning_mode = tuning_mode
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

class DummyModel(PTuneMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(10, config.hidden_size)
        self.init_prompts(config)

class TestPTuneMixin(unittest.TestCase):
    def test_ptune_shapes(self):
        batch_size = 2
        pre_seq_len = 5
        hidden_size = 16
        num_hidden_layers = 4

        config = DummyConfig(
            tuning_mode="ptune",
            pre_seq_len=pre_seq_len,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # In ptune mode, prompts shape is (batch_size, pre_seq_len, hidden_size)
        self.assertEqual(prompts.shape, (batch_size, pre_seq_len, hidden_size))

        # intermediate_prompts should be DUMMY
        self.assertTrue(intermediate_prompts is DUMMY or torch.allclose(intermediate_prompts, DUMMY))

    def test_deep_ptune_shapes(self):
        batch_size = 2
        pre_seq_len = 5
        hidden_size = 16
        num_hidden_layers = 4

        config = DummyConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=pre_seq_len,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers
        )

        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size)

        # In deep_ptune mode, prompts shape is (batch_size, pre_seq_len, hidden_size)
        self.assertEqual(prompts.shape, (batch_size, pre_seq_len, hidden_size))

        # intermediate_prompts shape should be (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        self.assertEqual(
            intermediate_prompts.shape,
            (num_hidden_layers, batch_size, pre_seq_len, hidden_size)
        )

        # Also check that the first layer (index 0) is all zeros
        self.assertTrue(torch.all(intermediate_prompts[0] == 0))

if __name__ == '__main__':
    unittest.main()
