import os
import sys

sys.path.insert(0, os.path.abspath('src'))
os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_intermediate_prompts_shape():
    from unittest import mock

    petals_utils_misc_mock = mock.MagicMock(__spec__=None)
    import torch
    petals_utils_misc_mock.DUMMY = torch.empty(0)

    import hivemind
    hivemind.get_logger = mock.MagicMock()
    hivemind.PeerID = mock.MagicMock()
    hivemind.MSGPackSerializer = mock.MagicMock()

    mocks = {
        'petals.utils.misc': petals_utils_misc_mock,
        'tensor_parallel': mock.MagicMock(__spec__=None),
        'petals.client.inference_session': mock.MagicMock(__spec__=None),
        'petals.client.remote_sequential': mock.MagicMock(__spec__=None),
        'petals.client.routing': mock.MagicMock(__spec__=None),
    }

    with mock.patch.dict('sys.modules', mocks):
        import torch.nn as nn
        from transformers import PretrainedConfig

        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            import petals.client.ptune as ptune

            class MockModel(ptune.PTuneMixin, nn.Module):
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

            model = MockModel(config)

            batch_size = 2
            prompts, intermediate_prompts = model.get_prompt(batch_size)

            assert prompts.shape == (batch_size, config.pre_seq_len, config.hidden_size)
            assert intermediate_prompts.shape == (config.num_hidden_layers - 1, batch_size, config.pre_seq_len, config.hidden_size)
