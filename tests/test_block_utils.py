import unittest
from unittest.mock import MagicMock, patch
import torch
from transformers import PretrainedConfig
from petals.server.block_utils import estimate_num_blocks
from petals.utils.convert_block import QuantType

class TestBlockUtils(unittest.TestCase):
    def setUp(self):
        self.block_config = PretrainedConfig(
            hidden_size=1024,
            num_hidden_layers=30,
            num_key_value_groups=1,
        )
        self.device = torch.device("cuda:0")
        self.torch_dtype = torch.float16
        self.quant_type = QuantType.NF4
        self.attn_cache_tokens = 4096

    @patch("petals.server.block_utils.get_block_size")
    @patch("torch.cuda.get_device_properties")
    def test_estimate_num_blocks_cuda(self, mock_get_device_properties, mock_get_block_size):
        # Mock GPU memory: 16 GB
        mock_get_device_properties.return_value.total_memory = 16 * 1024**3

        # Mock block size: 1 GB
        mock_get_block_size.return_value = 1 * 1024**3

        # Call function
        num_blocks = estimate_num_blocks(
            self.block_config,
            self.device,
            self.torch_dtype,
            self.quant_type,
            [],
            self.attn_cache_tokens,
        )

        self.assertTrue(num_blocks > 0)
        self.assertTrue(num_blocks <= 30)

    @patch("petals.server.block_utils.get_block_size")
    @patch("psutil.virtual_memory")
    def test_estimate_num_blocks_cpu(self, mock_virtual_memory, mock_get_block_size):
        # Mock CPU memory: 32 GB
        mock_virtual_memory.return_value.total = 32 * 1024**3

        mock_get_block_size.return_value = 1 * 1024**3

        device = torch.device("cpu")

        with self.assertRaises(AssertionError):
            estimate_num_blocks(
                self.block_config,
                device,
                torch.float32,
                QuantType.NONE,
                [],
                self.attn_cache_tokens,
            )
