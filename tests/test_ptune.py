import sys
import torch
from unittest import mock
import importlib.util

def test_ptune_intermediate_prompt_embeddings_shape():
    """
    Test that intermediate_prompt_embeddings have the correct shape
    (num_hidden_layers - 1) instead of num_hidden_layers.
    """
    petals_client_mock = mock.MagicMock(__path__=["src/petals/client"], __spec__=None)

    hivemind_mock = mock.MagicMock()
    tensor_parallel_mock = mock.MagicMock()

    with mock.patch.dict('sys.modules', {
        'hivemind': hivemind_mock,
        'tensor_parallel': tensor_parallel_mock,
        'petals': mock.MagicMock(),  # Mock petals to avoid deep recursive imports
    }):
        # Setup DUMMY correctly before loading ptune module
        sys.modules["petals.utils.misc"] = mock.MagicMock()
        sys.modules["petals.utils.misc"].DUMMY = torch.empty(0)

        # Import ptune module using spec
        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        ptune.DUMMY = torch.empty(0)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class MockConfig:
            def __init__(self):
                self.tuning_mode = "deep_ptune"
                self.pre_seq_len = 5
                self.hidden_size = 10
                self.num_hidden_layers = 4

        class TestModel(ptune.PTuneMixin):
            def __init__(self):
                self.word_embeddings = mock.MagicMock()
                self.word_embeddings.weight.device = torch.device('cpu')
                self.word_embeddings.weight.dtype = torch.float32
                self.config = MockConfig()

        model = TestModel()

        # Test init_prompts
        model.init_prompts(model.config)

        expected_dim = (model.config.num_hidden_layers - 1) * model.config.hidden_size
        assert model.intermediate_prompt_embeddings.embedding_dim == expected_dim, \
            f"Expected {expected_dim}, got {model.intermediate_prompt_embeddings.embedding_dim}"

        # Test get_prompt
        prompts, intermediate = model.get_prompt(2)

        expected_prompts_shape = (2, model.config.pre_seq_len, model.config.hidden_size)
        assert prompts.shape == expected_prompts_shape, \
            f"Expected {expected_prompts_shape}, got {prompts.shape}"

        expected_intermediate_shape = (
            model.config.num_hidden_layers - 1,
            2,
            model.config.pre_seq_len,
            model.config.hidden_size
        )
        assert intermediate.shape == expected_intermediate_shape, \
            f"Expected {expected_intermediate_shape}, got {intermediate.shape}"
