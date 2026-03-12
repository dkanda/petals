import pytest
import torch

from petals.utils.misc import is_dummy, docstring_from

def test_is_dummy():
    # Test with standard empty tensor
    empty_tensor = torch.empty(0)
    assert is_dummy(empty_tensor) is True

    # Test with non-empty tensor
    non_empty_tensor = torch.tensor([1, 2, 3])
    assert is_dummy(non_empty_tensor) is False

    # Additional checks for dummy constants
    from petals.utils.misc import DUMMY, DUMMY_INT64, DUMMY_KEY_PAST
    assert is_dummy(DUMMY) is True
    assert is_dummy(DUMMY_INT64) is True
    assert is_dummy(DUMMY_KEY_PAST) is True

def test_docstring_from():
    def source_func():
        """This is a source docstring."""
        pass

    @docstring_from(source_func)
    def dest_func():
        pass

    assert dest_func.__doc__ == "This is a source docstring."

    def another_source_func():
        """Another source docstring."""
        pass

    def another_dest_func():
        """This should be overwritten."""
        pass

    # Apply as a function rather than a decorator
    docstring_from(another_source_func)(another_dest_func)
    assert another_dest_func.__doc__ == "Another source docstring."
