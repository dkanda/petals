import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import yaml

from petals.cli.configure import run_wizard


class TestConfigure(unittest.TestCase):
    @patch("builtins.input")
    @patch("petals.cli.configure.estimate_num_blocks")
    @patch("petals.cli.configure.AutoDistributedConfig.from_pretrained")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    @patch("builtins.open", new_callable=mock_open)
    def test_wizard_defaults(
        self,
        mock_file,
        mock_cuda_available,
        mock_device_count,
        mock_get_device_properties,
        mock_config,
        mock_estimate,
        mock_input,
    ):
        # Mock inputs:
        # 1. Model: default (enter)
        # 2. Swarm: default (enter) -> public
        # 3. GPU: default (enter) -> all
        # 4. Public name: default (enter) -> None
        # 5. Token: default (enter) -> None
        mock_input.side_effect = ["", "", "", "", ""]

        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_device_properties.return_value.name = "Test GPU"
        mock_get_device_properties.return_value.total_memory = 16 * 1024**3
        mock_estimate.return_value = 5

        # Mock config attributes
        mock_config.return_value.hidden_size = 1024
        mock_config.return_value.num_hidden_layers = 10
        mock_config.return_value.num_key_value_groups = 1

        with patch("sys.stdout", new=MagicMock()):
            run_wizard()

        # Check if config.yml was written
        mock_file.assert_called_with("config.yml", "w")
        handle = mock_file()

        # Verify content written
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        data = yaml.safe_load(written_content)

        self.assertEqual(data["model"], "meta-llama/Meta-Llama-3.1-405B-Instruct")
        self.assertEqual(data["num_blocks"], 5)
        # initial_peers should be present for public swarm
        self.assertTrue(len(data["initial_peers"]) > 0)

    @patch("builtins.input")
    @patch("petals.cli.configure.estimate_num_blocks")
    @patch("petals.cli.configure.AutoDistributedConfig.from_pretrained")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    @patch("builtins.open", new_callable=mock_open)
    def test_wizard_custom_private(
        self,
        mock_file,
        mock_cuda_available,
        mock_device_count,
        mock_get_device_properties,
        mock_config,
        mock_estimate,
        mock_input,
    ):
        # Mock inputs:
        # 1. Model: custom (0) -> "my/model"
        # 2. Swarm: private (2) -> new (a)
        # 3. GPU: default (enter) -> all
        # 4. Public name: "MyServer"
        # 5. Token: "secret"
        mock_input.side_effect = ["0", "my/model", "2", "a", "", "MyServer", "secret"]

        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_device_properties.return_value.name = "Test GPU"
        mock_get_device_properties.return_value.total_memory = 16 * 1024**3
        mock_estimate.return_value = 10

        mock_config.return_value.hidden_size = 1024
        mock_config.return_value.num_hidden_layers = 10
        mock_config.return_value.num_key_value_groups = 1

        with patch("sys.stdout", new=MagicMock()):
            run_wizard()

        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        data = yaml.safe_load(written_content)

        self.assertEqual(data["model"], "my/model")
        self.assertEqual(data["public_name"], "MyServer")
        self.assertEqual(data["token"], "secret")
        self.assertTrue(data["new_swarm"])
        self.assertNotIn("initial_peers", data)
