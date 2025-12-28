import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import requests
import torch
import transformers
from hivemind import nested_compare, nested_flatten
from hivemind.p2p import PeerID

from petals import AutoDistributedConfig
from petals.models.gemma3 import DistributedGemma3Config
from petals.server.reachability import validate_reachability
from petals.server.throughput import measure_compute_rps
from petals.utils.convert_block import QuantType
from petals.utils.misc import DUMMY, is_dummy
from petals.utils.packaging import pack_args_kwargs, unpack_args_kwargs

os.environ["INITIAL_PEERS"] = "dummy_peer"
from test_utils import MODEL_NAME


def test_bnb_not_imported_when_unnecessary():
    """
    We avoid importing bitsandbytes when it's not used,
    since bitsandbytes doesn't always find correct CUDA libs and may raise exceptions because of that.

    If this test fails, please change your code to import bitsandbytes and/or petals.utils.peft
    in the function's/method's code when it's actually needed instead of importing them in the beginning of the file.
    This won't slow down the code - importing a module for the 2nd time doesn't rerun module code.
    """

    subprocess.check_call([sys.executable, "-c", "import petals, sys; assert 'bitsandbytes' not in sys.modules"])


@pytest.mark.forked
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("n_tokens", [1, 16])
@pytest.mark.parametrize("tensor_parallel", [False, True])
def test_compute_throughput(inference: bool, n_tokens: int, tensor_parallel: bool, monkeypatch):
    monkeypatch.setenv("INITIAL_PEERS", "dummy_peer")
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME)
    if tensor_parallel and config.model_type != "bloom":
        pytest.skip("Tensor parallelism is implemented only for BLOOM for now")

    tensor_parallel_devices = ("cpu", "cpu") if tensor_parallel else ()
    compute_rps = measure_compute_rps(
        config,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        quant_type=QuantType.NONE,
        tensor_parallel_devices=tensor_parallel_devices,
        n_tokens=n_tokens,
        n_steps=5,
        inference=inference,
    )
    assert isinstance(compute_rps, float) and compute_rps > 0


@pytest.mark.forked
def test_pack_inputs(monkeypatch):
    monkeypatch.setenv("INITIAL_PEERS", "dummy_peer")
    x = torch.ones(3)
    y = torch.arange(5)
    z = DUMMY

    args = (x, z, None, (y, y), z)
    kwargs = dict(foo=torch.zeros(1, 1), bar={"l": "i", "g": "h", "t": ("y", "e", "a", "r", torch.rand(1), x, y)})

    flat_tensors, args_structure = pack_args_kwargs(*args, **kwargs)

    assert len(flat_tensors) == 5
    assert all(isinstance(t, torch.Tensor) for t in flat_tensors)

    restored_args, restored_kwargs = unpack_args_kwargs(flat_tensors, args_structure)

    assert len(restored_args) == len(args)
    assert torch.all(restored_args[0] == x).item() and restored_args[2] is None
    assert nested_compare((args, kwargs), (restored_args, restored_kwargs))
    for original, restored in zip(nested_flatten((args, kwargs)), nested_flatten((restored_args, restored_kwargs))):
        if isinstance(original, torch.Tensor):
            assert torch.all(original == restored)
        else:
            assert original == restored


def test_validate_reachability(monkeypatch):
    monkeypatch.setenv("PETALS_IGNORE_DEPENDENCY_VERSIONS", "1")
    monkeypatch.setenv("INITIAL_PEERS", "dummy_peer")

    peer_id = PeerID.from_base58("Qme6XdVTfK4BGN8gD2qt93J2SGsSkCo35aTzSJ5T16u3xe")

    class MockSuccess:
        def json(self):
            return {"success": True}

        def raise_for_status(self):
            pass

    with monkeypatch.context() as m:
        m.setattr(requests, "get", lambda *args, **kwargs: MockSuccess())
        validate_reachability(peer_id, wait_time=0.1, retry_delay=0.05)

    with monkeypatch.context() as m:
        m.setattr(
            requests,
            "get",
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ConnectionError),
        )
        with pytest.raises(RuntimeError, match=r"Could not check server reachability"):
            validate_reachability(peer_id, wait_time=0.1, retry_delay=0.05)


def test_gemma_3_config(monkeypatch):
    monkeypatch.setenv("INITIAL_PEERS", "dummy_peer")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "gemma2"}, f)

        config = AutoDistributedConfig.from_pretrained(tmpdir)
        assert isinstance(config, DistributedGemma3Config)
        assert config.dht_prefix == f"{Path(tmpdir).name}-hf"

        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": "gemma2",
                    "quantization_config": {"load_in_8bit": True},
                },
                f,
            )
        config = AutoDistributedConfig.from_pretrained(tmpdir)
        assert not hasattr(config, "quantization_config")
