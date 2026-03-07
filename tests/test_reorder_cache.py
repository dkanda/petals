import pytest
import torch
from petals.models.block_utils import ReorderCacheMixin

class DummySelfAttn:
    def __init__(self, num_key_value_heads, q_head_dim=None, v_head_dim=None, head_dim=None):
        self.num_key_value_heads = num_key_value_heads
        if q_head_dim is not None:
            self.q_head_dim = q_head_dim
        if v_head_dim is not None:
            self.v_head_dim = v_head_dim
        if head_dim is not None:
            self.head_dim = head_dim

class DummyBlock(ReorderCacheMixin):
    def __init__(self, self_attn):
        self.self_attn = self_attn

def test_reorder_cache_deepseek():
    # DeepSeek uses q_head_dim and v_head_dim
    batch_size = 2
    num_key_value_heads = 4
    seq_length = 5
    q_head_dim = 128
    v_head_dim = 64

    dummy_attn = DummySelfAttn(num_key_value_heads, q_head_dim=q_head_dim, v_head_dim=v_head_dim)
    block = DummyBlock(dummy_attn)

    # test to_bloom
    key_states = torch.randn(batch_size, num_key_value_heads, seq_length, q_head_dim)
    value_states = torch.randn(batch_size, num_key_value_heads, seq_length, v_head_dim)

    # from_bloom to original: key_states is [batch_size, num_key_value_heads, seq_length, q_head_dim]
    # to_bloom expects:
    # key_states: [batch_size, num_key_value_heads, seq_length, q_head_dim] -> permuted to [batch_size * num_key_value_heads, q_head_dim, seq_length] ? Wait, original code:

    # Original reorder_cache_to_bloom:
    # value_states = value_states.view(batch_size * num_key_value_heads, seq_length, v_head_dim)
    # key_states = key_states.view(batch_size * num_key_value_heads, seq_length, q_head_dim)
    # key_states = key_states.permute(0, 2, 1) # [batch_size * num_key_value_heads, q_head_dim, seq_length]

    # Let's test the round trip
    to_bloom_key, to_bloom_value = block._reorder_cache_to_bloom((key_states, value_states), batch_size, seq_length)

    assert to_bloom_key.shape == (batch_size * num_key_value_heads, q_head_dim, seq_length)
    assert to_bloom_value.shape == (batch_size * num_key_value_heads, seq_length, v_head_dim)

    from_bloom_key, from_bloom_value = block._reorder_cache_from_bloom((to_bloom_key, to_bloom_value), batch_size, seq_length)

    assert from_bloom_key.shape == (batch_size, num_key_value_heads, seq_length, q_head_dim)
    assert from_bloom_value.shape == (batch_size, num_key_value_heads, seq_length, v_head_dim)

    assert torch.allclose(from_bloom_key, key_states)
    assert torch.allclose(from_bloom_value, value_states)

def test_reorder_cache_llama():
    # Llama uses head_dim
    batch_size = 2
    num_key_value_heads = 4
    seq_length = 5
    head_dim = 128

    dummy_attn = DummySelfAttn(num_key_value_heads, head_dim=head_dim)
    block = DummyBlock(dummy_attn)

    key_states = torch.randn(batch_size, num_key_value_heads, seq_length, head_dim)
    value_states = torch.randn(batch_size, num_key_value_heads, seq_length, head_dim)

    to_bloom_key, to_bloom_value = block._reorder_cache_to_bloom((key_states, value_states), batch_size, seq_length)

    assert to_bloom_key.shape == (batch_size * num_key_value_heads, head_dim, seq_length)
    assert to_bloom_value.shape == (batch_size * num_key_value_heads, seq_length, head_dim)

    from_bloom_key, from_bloom_value = block._reorder_cache_from_bloom((to_bloom_key, to_bloom_value), batch_size, seq_length)

    assert from_bloom_key.shape == (batch_size, num_key_value_heads, seq_length, head_dim)
    assert from_bloom_value.shape == (batch_size, num_key_value_heads, seq_length, head_dim)

    assert torch.allclose(from_bloom_key, key_states)
    assert torch.allclose(from_bloom_value, value_states)
