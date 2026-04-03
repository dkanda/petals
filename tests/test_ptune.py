import sys
from unittest import mock
import torch
import torch.nn as nn
from transformers import PretrainedConfig

def test_ptune_shapes_and_padding():
    with mock.patch.dict(sys.modules, {
        'hivemind': mock.MagicMock(),
        'petals': mock.MagicMock(),
        'petals.utils': mock.MagicMock(),
        'petals.utils.misc': mock.MagicMock(DUMMY="DUMMY_VAR")
    }):
        # Import the module dynamically inside the test context to avoid global pollution
        import importlib.util
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptune)
        PTuneMixin = ptune.PTuneMixin

        class DummyWordEmbeddings:
            def __init__(self):
                self.weight = torch.zeros(1, dtype=torch.float16)

        class DummyModel(PTuneMixin, nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.word_embeddings = DummyWordEmbeddings()
                self.init_prompts(config)

        pre_seq_len = 10
        hidden_size = 32
        num_hidden_layers = 4
        batch_size = 2

        config = PretrainedConfig(
            tuning_mode="deep_ptune",
            pre_seq_len=pre_seq_len,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers
        )

        model = DummyModel(config)

        # Verify parameter shapes
        assert model.prompt_embeddings.weight.shape == (pre_seq_len, hidden_size)
        assert model.intermediate_prompt_embeddings.weight.shape == (pre_seq_len, (num_hidden_layers - 1) * hidden_size)

        prompts, intermediate_prompts = model.get_prompt(batch_size=batch_size)

        # Verify outputs match embedding dtype
        assert prompts.dtype == torch.float16
        assert intermediate_prompts.dtype == torch.float16

        # Verify outputs shapes
        assert prompts.shape == (batch_size, pre_seq_len, hidden_size)
        assert intermediate_prompts.shape == (num_hidden_layers, batch_size, pre_seq_len, hidden_size)

        # Verify zero-padding in the first layer
        zero_padding = intermediate_prompts[0]
        assert torch.all(zero_padding == 0)

        # Verify intermediate layer prompts are not completely zeroed
        non_zero_padding = intermediate_prompts[1:]
        assert not torch.all(non_zero_padding == 0)
