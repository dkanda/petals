import pytest

from petals.data_structures import ServerInfo, ServerState


def test_server_info_from_tuple():
    # Valid tuple with a known extra field
    info = ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {"public_name": "test_server"}))
    assert info.state == ServerState.ONLINE
    assert info.throughput == 10.0
    assert info.public_name == "test_server"

    # Tuple with an unknown extra field should raise a ValueError
    with pytest.raises(ValueError, match=r"Unknown fields in ServerInfo: \['foo'\]"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {"foo": "bar"}))

    # Tuple with multiple unknown extra fields should raise a ValueError
    with pytest.raises(ValueError, match=r"Unknown fields in ServerInfo: \['bar', 'foo'\]"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {"foo": "baz", "bar": "qux"}))

    # Malformed tuples that should raise errors
    with pytest.raises(TypeError, match="info must be a tuple"):
        ServerInfo.from_tuple("not a tuple")

    with pytest.raises(ValueError, match=r"info tuple must have 2 or 3 elements, but got 1"):
        ServerInfo.from_tuple((ServerState.ONLINE.value,))

    with pytest.raises(ValueError, match=r"info tuple must have 2 or 3 elements, but got 4"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {}, "extra_element"))

    with pytest.raises(TypeError, match="info\\[0\\] must be an int"):
        ServerInfo.from_tuple(("invalid state", 10.0, {}))

    with pytest.raises(TypeError, match="info\\[1\\] must be a float or int"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, "invalid throughput", {}))

    with pytest.raises(TypeError, match="info\\[2\\] must be a dict"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, "not a dict"))

    with pytest.raises(ValueError, match="Invalid server state: 999"):
        ServerInfo.from_tuple((999, 10.0, {}))
