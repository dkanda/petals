import pytest

from petals.utils.random import sample_up_to

def test_sample_up_to_population_greater_than_k():
    population = [1, 2, 3, 4, 5]
    k = 3
    result = sample_up_to(population, k)
    assert len(result) == k
    assert all(item in population for item in result)

def test_sample_up_to_population_less_than_k():
    population = [1, 2, 3]
    k = 5
    result = sample_up_to(population, k)
    assert len(result) == len(population)
    assert result == population

def test_sample_up_to_k_zero():
    population = [1, 2, 3]
    k = 0
    result = sample_up_to(population, k)
    assert len(result) == 0
    assert result == []

def test_sample_up_to_empty_population():
    population = []
    k = 5
    result = sample_up_to(population, k)
    assert len(result) == 0
    assert result == []

def test_sample_up_to_non_list_collection():
    population_set = {1, 2, 3, 4, 5}
    k = 3
    result = sample_up_to(population_set, k)
    assert isinstance(result, list)
    assert len(result) == k
    assert all(item in population_set for item in result)

    population_tuple = (1, 2, 3)
    k_tuple = 5
    result_tuple = sample_up_to(population_tuple, k_tuple)
    assert isinstance(result_tuple, list)
    assert len(result_tuple) == len(population_tuple)
    assert result_tuple == list(population_tuple)
