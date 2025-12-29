import subprocess
import sys

import pytest
import requests
import torch
from hivemind import nested_compare, nested_flatten
from hivemind.p2p import PeerID

from petals import AutoDistributedConfig
from petals.data_structures import ServerInfo, ServerState
from petals.server.reachability import validate_reachability
from petals.server.throughput import measure_compute_rps
from petals.utils.convert_block import QuantType
from petals.utils.misc import DUMMY, is_dummy
from petals.utils.packaging import pack_args_kwargs, unpack_args_kwargs
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
def test_compute_throughput(inference: bool, n_tokens: int, tensor_parallel: bool):
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
def test_pack_inputs():
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


def test_server_info_from_tuple():
    valid_info = ServerInfo.from_tuple((ServerState.ONLINE.value, 100.0, {"version": "2.1.0"}))
    assert valid_info.state == ServerState.ONLINE
    assert valid_info.throughput == 100.0
    assert valid_info.version == "2.1.0"

    # Test with invalid data
    with pytest.raises(ValueError, match="expected a sequence of at least 2 elements"):
        ServerInfo.from_tuple(None)
    with pytest.raises(ValueError, match="expected a sequence of at least 2 elements"):
        ServerInfo.from_tuple((1,))
    with pytest.raises(ValueError, match="expected a dict of extra info"):
        ServerInfo.from_tuple((1, 100.0, "not-a-dict"))
