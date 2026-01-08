import pytest

from petals.data_structures import ServerInfo, ServerState


def test_server_info_from_tuple():
    # Test valid tuples
    info = ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0))
    assert info.state == ServerState.ONLINE
    assert info.throughput == 10.0

    info = ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {"start_block": 0, "end_block": 10}))
    assert info.state == ServerState.ONLINE
    assert info.throughput == 10.0
    assert info.start_block == 0
    assert info.end_block == 10

    # Test invalid tuples
    with pytest.raises(ValueError, match=r"info must have 2 or 3 elements, but got 1"):
        ServerInfo.from_tuple((ServerState.ONLINE.value,))

    with pytest.raises(ValueError, match=r"info must have 2 or 3 elements, but got 4"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {}, None))
