import sys
import unittest.mock as mock
import importlib.util
import torch

class MockPackage(mock.MagicMock):
    __path__ = []
    __spec__ = None

def test_deep_ptune_shapes():
    with mock.patch.dict("sys.modules", {
        "petals": MockPackage(),
        "petals.utils": MockPackage(),
        "petals.utils.misc": MockPackage(),
        "hivemind": MockPackage(),
        "transformers": MockPackage(),
    }):
        sys.modules["petals.utils.misc"].DUMMY = torch.zeros(1)

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyConfig:
            tuning_mode = "deep_ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 4

        class DummyPTune(ptune.PTuneMixin):
            def __init__(self):
                self.config = DummyConfig()

                class DummyWordEmbeddings:
                    weight = torch.zeros(1, dtype=torch.float32)
                self.word_embeddings = DummyWordEmbeddings()

                self.init_prompts(self.config)

        model = DummyPTune()

        assert model.intermediate_prompt_embeddings.weight.shape == (5, 48)

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert intermediate_prompts.shape == (4, 2, 5, 16)

        # Check that the first layer's prompt in intermediate_prompts is zeros
        assert torch.all(intermediate_prompts[0] == 0)

def test_ptune_shapes():
    with mock.patch.dict("sys.modules", {
        "petals": MockPackage(),
        "petals.utils": MockPackage(),
        "petals.utils.misc": MockPackage(),
        "hivemind": MockPackage(),
        "transformers": MockPackage(),
    }):
        sys.modules["petals.utils.misc"].DUMMY = torch.zeros(1)

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyConfig:
            tuning_mode = "ptune"
            pre_seq_len = 5
            hidden_size = 16
            num_hidden_layers = 4

        class DummyPTune(ptune.PTuneMixin):
            def __init__(self):
                self.config = DummyConfig()

                class DummyWordEmbeddings:
                    weight = torch.zeros(1, dtype=torch.float32)
                self.word_embeddings = DummyWordEmbeddings()

                self.init_prompts(self.config)

        model = DummyPTune()

        assert not hasattr(model, "intermediate_prompt_embeddings")

        prompts, intermediate_prompts = model.get_prompt(batch_size=2)

        assert prompts.shape == (2, 5, 16)
        assert "DUMMY" in repr(intermediate_prompts) or intermediate_prompts is sys.modules["petals.utils.misc"].DUMMY or (isinstance(intermediate_prompts, torch.Tensor) and intermediate_prompts.shape == (1,))
