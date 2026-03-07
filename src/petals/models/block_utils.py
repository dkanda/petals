from typing import Tuple
import torch

class ReorderCacheMixin:
    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)

        q_head_dim = getattr(self.self_attn, "q_head_dim", getattr(self.self_attn, "head_dim", None))
        v_head_dim = getattr(self.self_attn, "v_head_dim", getattr(self.self_attn, "head_dim", None))

        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, q_head_dim
        )
        value_states = value_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, v_head_dim
        )
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value

        q_head_dim = getattr(self.self_attn, "q_head_dim", getattr(self.self_attn, "head_dim", None))
        v_head_dim = getattr(self.self_attn, "v_head_dim", getattr(self.self_attn, "head_dim", None))

        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, v_head_dim
        )
        key_states = key_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, q_head_dim
        )
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
