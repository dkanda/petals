import sys
import os
from unittest import mock
import torch

sys.path.insert(0, os.path.abspath('src'))

def test_ptune_mixin_intermediate_prompts_shape():
    # Because we're isolating a submodule nested deep inside petals.client,
    # and the parent petals.__init__.py attempts to aggressively import
    # almost everything (including data_structures which imports hivemind.moe.expert_uid),
    # the cleanest standard-python way to test without cascading is to bypass
    # the parent initialization using mock.patch.dict on sys.modules,
    # as recommended in memory constraints when imports fail due to unresolvable dependencies.

    mock_hivemind = mock.MagicMock()
    mock_hivemind.dht = mock.MagicMock()
    mock_hivemind.p2p = mock.MagicMock()
    mock_hivemind.utils = mock.MagicMock()
    mock_hivemind.utils.logging = mock.MagicMock()
    mock_hivemind.moe = mock.MagicMock()
    mock_hivemind.moe.expert_uid = mock.MagicMock()
    mock_hivemind.proto = mock.MagicMock()

    # Polyfill top-level hivemind exports commonly used by petals
    mock_hivemind.PeerID = mock_hivemind.p2p.PeerID
    mock_hivemind.MSGPackSerializer = mock_hivemind.utils.MSGPackSerializer
    mock_hivemind.get_logger = mock_hivemind.utils.get_logger

    petals_mock = mock.MagicMock(__path__=["src/petals"], __spec__=None)
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)
    petals_mock.client = petals_client_mock

    with mock.patch.dict(
        "sys.modules",
        {
            "hivemind": mock_hivemind,
            "hivemind.dht": mock_hivemind.dht,
            "hivemind.p2p": mock_hivemind.p2p,
            "hivemind.utils": mock_hivemind.utils,
            "hivemind.utils.logging": mock_hivemind.utils.logging,
            "hivemind.moe": mock_hivemind.moe,
            "hivemind.moe.expert_uid": mock_hivemind.moe.expert_uid,
            "hivemind.proto": mock_hivemind.proto,
            "tensor_parallel": mock.MagicMock(),

            # The crucial fix: Block the parent __init__ cascades
            "petals": petals_mock,
            "petals.client": petals_client_mock,

            # Mock heavy downstream client modules
            "petals.client.inference_session": mock.MagicMock(),
            "petals.client.remote_sequential": mock.MagicMock(),
            "petals.client.routing": mock.MagicMock(),
        }
    ):
        # Now we can import the module correctly using standard import
        import petals.client.ptune as ptune

        # Explicitly assign dummy constant since we bypassed parent init
        ptune.DUMMY = torch.empty(0)

        class MockConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 10
            hidden_size = 128
            num_hidden_layers = 12

        class MockMixin(ptune.PTuneMixin):
            def __init__(self):
                self.config = MockConfig()
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = "cpu"
                self.word_embeddings.weight.dtype = torch.float32

        mixin = MockMixin()
        mixin.init_prompts(mixin.config)

        prompts, intermediate_prompts = mixin.get_prompt(batch_size=2)

        assert prompts.shape == torch.Size([2, 10, 128])
        # The key assertion: intermediate prompts must be num_hidden_layers - 1 (11)
        assert intermediate_prompts.shape == torch.Size([11, 2, 10, 128])
