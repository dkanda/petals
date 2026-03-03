import unittest
from unittest.mock import patch
import sys
from io import StringIO

from petals.cli.check_network import main

class TestCheckNetworkCLI(unittest.TestCase):
    @patch("petals.cli.check_network.check_direct_reachability")
    @patch("sys.argv", ["check_network.py", "--port", "31337"])
    def test_network_reachable(self, mock_check):
        mock_check.return_value = True

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            main()
        except SystemExit as e:
            self.fail(f"main() exited unexpectedly with code {e.code}")
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("Checking reachability on port 31337", output)
        self.assertIn("Success! Your node is reachable", output)
        mock_check.assert_called_once_with(port=31337, host_maddrs=["/ip4/0.0.0.0/tcp/31337"])

    @patch("petals.cli.check_network.check_direct_reachability")
    @patch("sys.argv", ["check_network.py", "--port", "31337"])
    def test_network_unreachable(self, mock_check):
        mock_check.return_value = False

        captured_output = StringIO()
        sys.stdout = captured_output

        with self.assertRaises(SystemExit) as cm:
            main()

        sys.stdout = sys.__stdout__

        self.assertEqual(cm.exception.code, 1)
        output = captured_output.getvalue()
        self.assertIn("Checking reachability on port 31337", output)
        self.assertIn("Failure! Your node is not reachable", output)
        self.assertIn("you need to set up port forwarding", output)
        mock_check.assert_called_once_with(port=31337, host_maddrs=["/ip4/0.0.0.0/tcp/31337"])

    @patch("petals.cli.check_network.check_direct_reachability")
    @patch("sys.argv", ["check_network.py", "--public_ip", "1.2.3.4", "--port", "31330"])
    def test_network_unreachable_with_ip(self, mock_check):
        mock_check.return_value = False

        captured_output = StringIO()
        sys.stdout = captured_output

        with self.assertRaises(SystemExit) as cm:
            main()

        sys.stdout = sys.__stdout__

        self.assertEqual(cm.exception.code, 1)
        output = captured_output.getvalue()
        self.assertIn("Checking reachability on port 31330", output)
        mock_check.assert_called_once_with(port=31330, host_maddrs=["/ip4/1.2.3.4/tcp/31330"])

if __name__ == "__main__":
    unittest.main()
