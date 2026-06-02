import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from unittest import mock
import torch

os.environ['PETALS_IGNORE_DEPENDENCY_VERSION'] = '1'

def test_ptune_fixed():
    with mock.patch.dict('sys.modules', {
        'hivemind': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.dht': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.p2p': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.utils': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.utils.logging': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.utils.tensor_deserializer': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.moe': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.moe.expert_uid': mock.MagicMock(__path__=[], __spec__=None),
        'hivemind.compression': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.inference_session': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.remote_sequential': mock.MagicMock(__path__=[], __spec__=None),
        'petals.client.routing': mock.MagicMock(__path__=[], __spec__=None),
    }):
        import hivemind
        hivemind.PeerID = mock.MagicMock()
        hivemind.MSGPackSerializer = mock.MagicMock()
        hivemind.get_logger = mock.MagicMock()

        with mock.patch('transformers.utils.import_utils.is_torch_fx_available', return_value=False, create=True):
            import petals.client.ptune

            with mock.patch.object(petals.client.ptune, '_original_register_parameter', torch.nn.Module.register_parameter):
                from transformers import PretrainedConfig

                class DummyWordEmbeddings:
                    def __init__(self):
                        self.weight = torch.empty(0, dtype=torch.float32)

                class TestPTuneMixin(petals.client.ptune.PTuneMixin):
                    def __init__(self, config):
                        self.config = config
                        self.word_embeddings = DummyWordEmbeddings()

                config = PretrainedConfig(
                    tuning_mode="deep_ptune",
                    pre_seq_len=16,
                    hidden_size=64,
                    num_hidden_layers=12
                )

                mixin = TestPTuneMixin(config)

                import petals.utils.misc
                petals.utils.misc.DUMMY = torch.empty(0)

                mixin.init_prompts(config)

                assert mixin.intermediate_prompt_embeddings.weight.shape == (16, 11 * 64)

                prompts, intermediate_prompts = mixin.get_prompt(batch_size=2)

                assert intermediate_prompts.shape == (11, 2, 16, 64)
