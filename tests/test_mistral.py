import pytest
import torch
from unittest.mock import patch, MagicMock

import hivemind
from petals.models.mistral import DistributedMistralForCausalLM, DistributedMistralConfig
from petals.client.remote_sequential import RemoteSequenceManager


@pytest.fixture
def model_name():
    return "mistralai/Devstral-Small-2507"


def test_model_sharding(model_name):
    hf_config = DistributedMistralConfig(
        model_type="mistral",
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=1,
        intermediate_size=1,
        vocab_size=1,
        dht_prefix="Devstral-Small-2507",
        initial_peers=["/ip4/127.0.0.1/tcp/31337"],
    )

    with patch("petals.client.remote_sequential.RemoteSequenceManager") as mock_sequence_manager:
        mock_sequence_manager.return_value = MagicMock(spec=RemoteSequenceManager)
        mock_dht = MagicMock(spec=hivemind.DHT)
        mock_dht.is_alive.return_value = True
        DistributedMistralForCausalLM(hf_config, dht=mock_dht)

        mock_sequence_manager.assert_called_once()
        args, kwargs = mock_sequence_manager.call_args
        assert kwargs["dht"] is mock_dht

        config = args[0]
        assert isinstance(config, DistributedMistralConfig)
        assert config.num_hidden_layers == 40
        assert config.dht_prefix == "Devstral-Small-2507"

        block_uids = args[1]
        expected_uids = tuple(f"Devstral-Small-2507.{i}" for i in range(40))
        assert block_uids == expected_uids
