import pytest
import importlib.util
from unittest import mock

def test_ptune():
    class MockTensor:
        def __init__(self, shape=None):
            self.shape = shape
        def long(self): return self
        def unsqueeze(self, dim):
            s = list(self.shape)
            s.insert(dim if dim >= 0 else len(s)+dim+1, 1)
            return MockTensor(shape=s)
        def expand(self, *args):
            return MockTensor(shape=[a if a != -1 else b for a, b in zip(args, self.shape)])
        def to(self, dtype_or_device): return self
        def view(self, *args): return MockTensor(shape=list(args))
        def permute(self, dims): return MockTensor(shape=[self.shape[i] for i in dims])
        def __call__(self, *args, **kwargs): return self
        def squeeze(self, *args, **kwargs): return self

    class MockTorch:
        float32 = 'float32'
        def arange(self, n): return MockTensor(shape=(n,))
        def cat(self, tensors, dim=0):
            s = list(tensors[0].shape)
            s[dim] += tensors[1].shape[dim]
            return MockTensor(shape=s)

    class MockNN:
        class Module:
            def register_parameter(self, *args, **kwargs): pass
        def Embedding(self, num_embeddings, embedding_dim, dtype=None):
            def _call(x, *args, **kwargs):
                return MockTensor(shape=list(x.shape) + [embedding_dim])
            return _call

    class MockConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def mock_logger(*args, **kwargs):
        return mock_logger
    mock_logger.info = lambda *args, **kwargs: None
    mock_logger.warning = lambda *args, **kwargs: None
    mock_logger.error = lambda *args, **kwargs: None

    class Dummy:
        def to(self, *args, **kwargs):
            return self

    import sys
    with mock.patch.dict(sys.modules):
        mock_torch = MockTorch()
        mock_torch.nn = MockNN()
        sys.modules['torch'] = mock_torch
        sys.modules['torch.nn'] = mock_torch.nn
        sys.modules['hivemind'] = type('MockHivemind', (), {'get_logger': mock_logger})
        sys.modules['transformers'] = type('MockTransformers', (), {'PretrainedConfig': MockConfig})
        sys.modules['petals.utils.misc'] = type('MockMisc', (), {'DUMMY': Dummy()})

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = type('MockEmbedding', (), {'weight': type('MockWeight', (), {'device': 'cpu', 'dtype': 'float32'})()})()
                self.init_prompts(config)

        config = MockConfig(hidden_size=64, num_hidden_layers=10, tuning_mode="ptune", pre_seq_len=8)
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == [2, 8, 64]
        assert isinstance(intermediate_prompts, Dummy)

def test_deep_ptune():
    class MockTensor:
        def __init__(self, shape=None):
            self.shape = shape
        def long(self): return self
        def unsqueeze(self, dim):
            s = list(self.shape)
            s.insert(dim if dim >= 0 else len(s)+dim+1, 1)
            return MockTensor(shape=s)
        def expand(self, *args):
            return MockTensor(shape=[a if a != -1 else b for a, b in zip(args, self.shape)])
        def to(self, dtype_or_device): return self
        def view(self, *args): return MockTensor(shape=list(args))
        def permute(self, dims): return MockTensor(shape=[self.shape[i] for i in dims])
        def __call__(self, *args, **kwargs): return self
        def squeeze(self, *args, **kwargs): return self

    class MockTorch:
        float32 = 'float32'
        def arange(self, n): return MockTensor(shape=(n,))
        def cat(self, tensors, dim=0):
            s = list(tensors[0].shape)
            s[dim] += tensors[1].shape[dim]
            return MockTensor(shape=s)

    class MockNN:
        class Module:
            def register_parameter(self, *args, **kwargs): pass
        def Embedding(self, num_embeddings, embedding_dim, dtype=None):
            def _call(x, *args, **kwargs):
                return MockTensor(shape=list(x.shape) + [embedding_dim])
            return _call

    class MockConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def mock_logger(*args, **kwargs):
        return mock_logger
    mock_logger.info = lambda *args, **kwargs: None
    mock_logger.warning = lambda *args, **kwargs: None
    mock_logger.error = lambda *args, **kwargs: None

    class Dummy:
        def to(self, *args, **kwargs):
            return self

    import sys
    with mock.patch.dict(sys.modules):
        mock_torch = MockTorch()
        mock_torch.nn = MockNN()
        sys.modules['torch'] = mock_torch
        sys.modules['torch.nn'] = mock_torch.nn
        sys.modules['hivemind'] = type('MockHivemind', (), {'get_logger': mock_logger})
        sys.modules['transformers'] = type('MockTransformers', (), {'PretrainedConfig': MockConfig})
        sys.modules['petals.utils.misc'] = type('MockMisc', (), {'DUMMY': Dummy()})

        spec = importlib.util.spec_from_file_location("ptune", "src/petals/client/ptune.py")
        ptune = importlib.util.module_from_spec(spec)
        sys.modules["petals.client.ptune"] = ptune
        spec.loader.exec_module(ptune)

        class DummyModel(ptune.PTuneMixin):
            def __init__(self, config):
                self.config = config
                self.word_embeddings = type('MockEmbedding', (), {'weight': type('MockWeight', (), {'device': 'cpu', 'dtype': 'float32'})()})()
                self.init_prompts(config)

        config = MockConfig(hidden_size=64, num_hidden_layers=10, tuning_mode="deep_ptune", pre_seq_len=8)
        model = DummyModel(config)
        prompts, intermediate_prompts = model.get_prompt(batch_size=2)
        assert prompts.shape == [2, 8, 64]
        assert intermediate_prompts.shape == [10, 2, 8, 64]
