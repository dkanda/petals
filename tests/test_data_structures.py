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

    with pytest.raises(ValueError, match=r"info must have exactly 3 elements \(got 1\)"):
        ServerInfo.from_tuple((ServerState.ONLINE.value,))

    with pytest.raises(ValueError, match=r"info must have exactly 3 elements \(got 2\)"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0))

    with pytest.raises(ValueError, match=r"info must have exactly 3 elements \(got 4\)"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, {}, "junk"))

    with pytest.raises(TypeError, match="info\\[0\\] must be an int"):
        ServerInfo.from_tuple(("invalid state", 10.0, {}))

    with pytest.raises(TypeError, match="info\\[1\\] must be a float or int"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, "invalid throughput", {}))

    with pytest.raises(TypeError, match="info\\[2\\] must be a dict"):
        ServerInfo.from_tuple((ServerState.ONLINE.value, 10.0, "not a dict"))

    with pytest.raises(ValueError, match="Invalid server state: 999"):
        ServerInfo.from_tuple((999, 10.0, {}))


def test_server_info_block_invariants():
    with pytest.raises(ValueError, match=r"start_block \(10\) must be less than end_block \(5\)"):
        ServerInfo(state=ServerState.ONLINE, throughput=10.0, start_block=10, end_block=5)

    with pytest.raises(ValueError, match=r"start_block \(10\) must be less than end_block \(10\)"):
        ServerInfo(state=ServerState.ONLINE, throughput=10.0, start_block=10, end_block=10)


def test_parse_uid():
    from petals.data_structures import parse_uid

    assert parse_uid("bloom.0") == ("bloom", 0)
    assert parse_uid("my.model.v1.10") == ("my.model.v1", 10)

    with pytest.raises(ValueError, match="does not support chained UIDs"):
        parse_uid("bloom.0 bloom.1")

    with pytest.raises(ValueError, match="expects UID in format"):
        parse_uid("invalid_uid")

    with pytest.raises(ValueError, match="expects index to be an integer"):
        parse_uid("bloom.abc")


def test_model_info_from_dict():
    from petals.data_structures import ModelInfo

    info = ModelInfo.from_dict({"num_blocks": 10, "repository": "test/repo"})
    assert info.num_blocks == 10
    assert info.repository == "test/repo"

    with pytest.raises(ValueError, match="Unknown fields in ModelInfo"):
        ModelInfo.from_dict({"num_blocks": 10, "foo": "bar"})
