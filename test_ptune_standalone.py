import torch
import torch.nn as nn
from transformers import PretrainedConfig

class DummyConfig(PretrainedConfig):
    def __init__(self):
        self.pre_seq_len = 10
        self.tuning_mode = "deep_ptune"
        self.hidden_size = 32
        self.num_hidden_layers = 4

class DummyModel:
    def __init__(self, config):
        self.config = config
        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prompt_embeddings = nn.Embedding(self.pre_seq_len, config.hidden_size, dtype=torch.float32)
        self.intermediate_prompt_embeddings = nn.Embedding(
            self.pre_seq_len,
            (config.num_hidden_layers - 1) * config.hidden_size,
            dtype=torch.float32,
        )
        self.word_embeddings = type('DummyEmbed', (), {'weight': torch.randn(10, config.hidden_size)})()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_tokens = prefix_tokens.to(self.word_embeddings.weight.device)
        prompts = self.prompt_embeddings(prefix_tokens)

        if self.config.tuning_mode == "deep_ptune":
            intermediate_prompts = self.intermediate_prompt_embeddings(prefix_tokens)
            intermediate_prompts = intermediate_prompts.view(
                batch_size,
                self.pre_seq_len,
                self.config.num_hidden_layers - 1,
                self.config.hidden_size,
            )
            intermediate_prompts = intermediate_prompts.permute([2, 0, 1, 3])
            intermediate_prompts = torch.cat([prompts.unsqueeze(0), intermediate_prompts], dim=0)
        else:
            intermediate_prompts = None

        dtype = self.word_embeddings.weight.dtype
        return prompts.to(dtype), intermediate_prompts.to(dtype)

config = DummyConfig()
model = DummyModel(config)
p, ip = model.get_prompt(2)
print("p.shape:", p.shape)
print("ip.shape:", ip.shape)
