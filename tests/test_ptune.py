import torch
import sys
import os
from unittest import mock

# Insert src directory to the python path to import modules without installing Petals
sys.path.insert(0, os.path.abspath('src'))

# Mock petals.utils.misc as needed
mock_petals_utils_misc = mock.MagicMock()
mock_petals_utils_misc.DUMMY = torch.empty(0)

with mock.patch.dict('sys.modules', {
    'petals.utils.misc': mock_petals_utils_misc,
    'hivemind': mock.MagicMock(),
}):
    # Import ptune directly to bypass hivemind and petals root package dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
    ptune = importlib.util.module_from_spec(spec)
    sys.modules["ptune"] = ptune
    spec.loader.exec_module(ptune)

    PTuneMixin = ptune.PTuneMixin
    from transformers import PretrainedConfig

def test_ptune_intermediate_prompts_shape():
    class TestPTuneMixin(PTuneMixin):
        def __init__(self, config):
            self.config = config
            # Create dummy word_embeddings parameter
            if hasattr(config, 'vocab_size'):
                self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
            else:
                self.word_embeddings = torch.nn.Embedding(100, config.hidden_size)

    # Initialize dummy config for deep_ptune
    config = PretrainedConfig(
        tuning_mode="deep_ptune",
        pre_seq_len=10,
        hidden_size=32,
        num_hidden_layers=5
    )

    model = TestPTuneMixin(config)
    model.init_prompts(config)

    # Check intermediate_prompt_embeddings parameter shape
    # Since num_hidden_layers=5, the param should have size 10 x ((5 - 1) * 32) = 10 x 128
    assert model.intermediate_prompt_embeddings.weight.shape == torch.Size([10, 128])

    # Check the result of get_prompt
    batch_size = 2
    prompts, intermediate_prompts = model.get_prompt(batch_size)

    # intermediate_prompts should be permuted to shape:
    # (num_hidden_layers - 1, batch_size, pre_seq_len, hidden_size)
    # i.e., (5 - 1, 2, 10, 32) -> (4, 2, 10, 32)
    assert intermediate_prompts.shape == torch.Size([4, 2, 10, 32])
