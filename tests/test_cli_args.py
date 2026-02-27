
import sys
import unittest
from unittest.mock import MagicMock, patch

from petals.cli.run_server import main


class TestRunServerCLI(unittest.TestCase):
    @patch("petals.cli.run_server.Server")
    def test_home_coordinator_mode(self, mock_server):
        test_args = [
            "petals-server",
            "--mode",
            "home-coordinator",
            "--converted_model_name_or_path",
            "bigscience/bloom-560m",
            "--device",
            "cpu",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that Server was initialized with new_swarm=True (implied by empty initial_peers)
        _, kwargs = mock_server.call_args
        self.assertEqual(kwargs["initial_peers"], [])

    @patch("petals.cli.run_server.Server")
    def test_home_worker_mode(self, mock_server):
        join_code = "/ip4/1.2.3.4/tcp/31337/p2p/Qm..."
        test_args = [
            "petals-server",
            "--mode",
            "home-worker",
            "--join",
            join_code,
            "--converted_model_name_or_path",
            "bigscience/bloom-560m",
            "--device",
            "cpu",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that Server was initialized with initial_peers=[join_code]
        _, kwargs = mock_server.call_args
        self.assertEqual(kwargs["initial_peers"], [join_code])

    @patch("petals.cli.run_server.Server")
    def test_default_mode(self, mock_server):
        test_args = [
            "petals-server",
            "--converted_model_name_or_path",
            "bigscience/bloom-560m",
            "--device",
            "cpu",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that Server was initialized with default initial_peers (public swarm)
        _, kwargs = mock_server.call_args
        # PUBLIC_INITIAL_PEERS is imported in run_server, so we can't easily check identity equality if it's not exposed
        # But we know it shouldn't be empty
        self.assertTrue(len(kwargs["initial_peers"]) > 0)

    @patch("petals.cli.run_server.Server")
    def test_worker_missing_join_code(self, mock_server):
        test_args = [
            "petals-server",
            "--mode",
            "home-worker",
            "--converted_model_name_or_path",
            "bigscience/bloom-560m",
        ]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                main()
