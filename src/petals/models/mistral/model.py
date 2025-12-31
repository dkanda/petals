from typing import Optional

import hivemind
import torch
from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.remote_generation import RemoteGenerationMixin
from petals.client.remote_sequential import RemoteSequential
from petals.models.mistral.config import DistributedMistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralModel, MistralPreTrainedModel


class DistributedMistralModel(FromPretrainedMixin, MistralModel):
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]

    def __init__(self, config: DistributedMistralConfig, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0
        super().__init__(config)
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _init_weights(self, module):
        pass


class DistributedMistralForCausalLM(RemoteGenerationMixin, FromPretrainedMixin, MistralForCausalLM):
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]

    def __init__(self, config: DistributedMistralConfig, *, dht: Optional[hivemind.DHT] = None):
        MistralPreTrainedModel.__init__(self, config)
        self.model = DistributedMistralModel(config, dht=dht)
        self.vocab_size = config.vocab_size
        self.lm_head = self.model.lm_head
        self.pre_seq_len = 0
        self.prefix_tuning = None

    @property
    def transformer(self):
        return self.model

    @property
    def word_embeddings(self):
        return self.model.embed_tokens

    @property
    def word_embeddings_layernorm(self):
        return self.model.norm

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def _init_weights(self, module):
        pass
