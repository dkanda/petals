import pytest
from petals.data_structures import parse_uid, UID_DELIMITER, CHAIN_DELIMITER

def test_parse_uid_valid():
    assert parse_uid("bloom.0") == ("bloom", 0)
    assert parse_uid("petals.bloom.123") == ("petals.bloom", 123)
    assert parse_uid("my.custom.prefix.42") == ("my.custom.prefix", 42)

def test_parse_uid_malformed():
    with pytest.raises(ValueError, match="Wrong UID format"):
        parse_uid("bloom")
    with pytest.raises(ValueError, match="index must be an integer"):
        parse_uid("bloom.")
    with pytest.raises(ValueError, match="index must be an integer"):
        parse_uid("bloom.notanint")

def test_parse_uid_chained():
    uid = f"bloom.0{CHAIN_DELIMITER}bloom.1"
    with pytest.raises(ValueError, match="does not support chained UIDs"):
        parse_uid(uid)
