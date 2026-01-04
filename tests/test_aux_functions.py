import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
import torch
from hivemind import nested_compare, nested_flatten
from hivemind.p2p import PeerID

from petals import AutoDistributedConfig
from petals.data_structures import ServerState
from petals.models.deepseek.config import DistributedDeepseekConfig
from petals.server.reachability import validate_reachability
from petals.server.throughput import measure_compute_rps
from petals.utils.convert_block import QuantType
from petals.utils.dht import _get_remote_module_infos
from petals.utils.misc import DUMMY, is_dummy
from petals.utils.packaging import pack_args_kwargs, unpack_args_kwargs
from test_utils import MODEL_NAME


@pytest.mark.asyncio
@patch("petals.utils.dht.PeerID")
async def test_get_remote_module_infos_with_corrupted_data(mock_peer_id):
    dht = SimpleNamespace(
        num_workers=1,
        run_coroutine=lambda coro, return_future: coro(None, dht),
    )

    async def _get_many(*args, **kwargs):
        return {
            "fake.uid.0": SimpleNamespace(
                value={
                    "peer1": SimpleNamespace(value=(ServerState.ONLINE.value, 1.0, {})),
                    "peer2": None,
                }
            )
        }

    node = SimpleNamespace(get_many=_get_many)

    infos = await _get_remote_module_infos(
        dht,
        node,
        ["fake.uid.0"],
        active_adapter=None,
        expiration_time=1234567890.0,
        latest=False,
    )
    assert len(infos) == 1
    assert len(infos[0].servers) == 1


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


@patch("petals.models.deepseek.config.DeepseekV3Config.from_pretrained")
def test_deepseek_dht_prefix(mock_from_pretrained):
    """
    This test ensures that the DHT prefix for DeepSeek models is correctly generated.
    The prefix should be derived from the model name, with slashes replaced by hyphens,
    and have a "-hf" suffix.
    """
    mock_config = MagicMock()
    mock_from_pretrained.return_value = mock_config

    result_config = DistributedDeepseekConfig.from_pretrained("deepseek-ai/deepseek-v3-large-base")

    # Check that the DHT prefix was correctly calculated and passed to the superclass method
    mock_from_pretrained.assert_called_once()
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs.get("dht_prefix") == "deepseek-v3-large-base-hf"
    assert result_config == mock_config


@patch("petals.models.deepseek.config.DeepseekV3Config.from_pretrained")
def test_deepseek_quantization_config_removal(mock_from_pretrained):
    """
    This test ensures that the `quantization_config` attribute is removed from the
    DeepSeek model configuration. This is necessary for compatibility with clients
    that do not support the model's native quantization.
    """
    mock_config = MagicMock()
    # Set the attribute we expect to be deleted
    mock_config.quantization_config = {"bits": 8}

    # from_pretrained should return the config object directly
    mock_from_pretrained.return_value = mock_config

    # The model name here is just a dummy since the call is mocked
    result_config = DistributedDeepseekConfig.from_pretrained("dummy-model")

    # Assert that the attribute was deleted
    assert not hasattr(mock_config, "quantization_config")
    # Also check we got the right object back
    assert result_config == mock_config
