import sys
import types
from unittest import mock

def test_ptune_shapes():
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    with open("src/petals/client/ptune.py", "r") as f:
        src_code = f.read()

    hivemind_mock = types.ModuleType("hivemind")
    hivemind_mock.get_logger = lambda name: mock.MagicMock()

    petals_utils_misc_mock = types.ModuleType("petals.utils.misc")
    petals_utils_misc_mock.DUMMY = None

    sys_modules_mocks = {
        "hivemind": hivemind_mock,
        "petals": types.ModuleType("petals"),
        "petals.utils": types.ModuleType("petals.utils"),
        "petals.utils.misc": petals_utils_misc_mock,
    }

    with mock.patch.dict("sys.modules", sys_modules_mocks):
        ptune = types.ModuleType("ptune")
        exec(src_code, ptune.__dict__)

        class Config(PretrainedConfig):
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 10
            num_hidden_layers = 3

        config = Config()

        mixin = ptune.PTuneMixin()
        mixin.word_embeddings = nn.Embedding(10, 10)
        mixin.config = config
        mixin.init_prompts(config)

        prompts, intermediate_prompts = mixin.get_prompt(2)

        assert prompts.shape == torch.Size([2, 5, 10])
        assert intermediate_prompts.shape == torch.Size([3, 2, 5, 10])
        assert torch.equal(intermediate_prompts[0], prompts)
