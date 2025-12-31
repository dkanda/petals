import pytest

from petals.data_structures import ServerInfo, ServerState


def test_server_info_from_tuple():
    # Valid tuple with a known extra field
    info = ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {"public_name": "test_server"}))
    assert info.state == ServerState.ONLINE
    assert info.throughput == 10.0
    assert info.public_name == "test_server"

    # Valid tuple with an unknown extra field (should be ignored)
    info_with_unknown = ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {"foo": "bar"}))
    assert not hasattr(info_with_unknown, "foo")

    # Malformed tuples that should raise errors
    with pytest.raises(TypeError, match="info must be a tuple"):
        ServerInfo.from_tuple("not a tuple")

    with pytest.raises(ValueError, match="info must have at least 2 elements"):
        ServerInfo.from_tuple((ServerState.ONLINE.value,))

    with pytest.raises(TypeError, match="info\\[0\\] must be an int"):
        ServerInfo.from_tuple(("invalid state", 10.0, {}))

    with pytest.raises(TypeError, match="info\\[1\\] must be a float or int"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, "invalid throughput", {}))

    with pytest.raises(TypeError, match="info\\[2\\] must be a dict"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, "not a dict"))

    with pytest.raises(ValueError, match="Invalid server state: 999"):
        ServerInfo.from_tuple((999, 10.0, {}))
