import pytest
from petals.data_structures import parse_uid

def test_parse_uid_valid():
    assert parse_uid("bigscience/bloom-560m.0") == ("bigscience/bloom-560m", 0)
    assert parse_uid("model.with.dots.123") == ("model.with.dots", 123)
    assert parse_uid("just_one_part.5") == ("just_one_part", 5)
    assert parse_uid("model.1.2") == ("model.1", 2)

def test_parse_uid_chained():
    with pytest.raises(ValueError, match=r"parse_uid\(\) does not support chained UIDs"):
        parse_uid("model.0 model.1")

def test_parse_uid_malformed():
    with pytest.raises(ValueError, match="Malformed module UID: no_dots_here"):
        parse_uid("no_dots_here")

    with pytest.raises(ValueError, match=r"Malformed module UID: \."):
        parse_uid(".")

    with pytest.raises(ValueError, match=r"Malformed module UID: prefix\."):
        parse_uid("prefix.")

    with pytest.raises(ValueError, match=r"Malformed module UID: \.index"):
        parse_uid(".index")

def test_parse_uid_non_integer_index():
    with pytest.raises(ValueError, match="Module index must be an integer, got not_an_int"):
        parse_uid("model.not_an_int")

    with pytest.raises(ValueError, match=r"Module index must be an integer, got 2a \(UID: model\.1\.2a\)"):
        parse_uid("model.1.2a")
