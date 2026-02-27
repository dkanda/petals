import sys
from unittest.mock import MagicMock, patch

import pytest
from petals.cli.run_server import main


def test_home_coordinator_mode():
    """Verify that home-coordinator mode forces new_swarm=True and prints the join code."""
    with patch("sys.argv", [
        "petals-server",
        "--mode", "home-coordinator",
        "bigscience/bloom-560m",
        "--device", "cpu",
        "--throughput", "dry_run"
    ]), \
    patch("petals.cli.run_server.Server") as MockServer, \
    patch("petals.cli.run_server.validate_version"), \
    patch("builtins.print") as mock_print:

        # Setup mock server
        mock_server_instance = MockServer.return_value
        mock_server_instance.dht.get_visible_maddrs.return_value = ["/ip4/1.2.3.4/tcp/31337/p2p/QmHash"]

        # Run main
        try:
            main()
        except SystemExit:
            pass

        # Check Server init args
        assert MockServer.call_count == 1
        call_args = MockServer.call_args[1]

        # new_swarm is popped before Server init, so we can't check it directly here.
        # But we can check initial_peers is empty, which happens if new_swarm was True.
        assert call_args["initial_peers"] == []

        # Check if join code was printed
        printed_text = "".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "HOME SWARM COORDINATOR STARTED" in printed_text
        assert "--mode home-worker --join /ip4/1.2.3.4/tcp/31337/p2p/QmHash" in printed_text


def test_home_worker_mode():
    """Verify that home-worker mode sets initial_peers correctly."""
    join_code = "/ip4/1.2.3.4/tcp/31337/p2p/QmHash"

    with patch("sys.argv", [
        "petals-server",
        "--mode", "home-worker",
        "--join", join_code,
        "bigscience/bloom-560m",
        "--device", "cpu",
        "--throughput", "dry_run"
    ]), \
    patch("petals.cli.run_server.Server") as MockServer, \
    patch("petals.cli.run_server.validate_version"):

        # Run main
        try:
            main()
        except SystemExit:
            pass

        # Check Server init args
        assert MockServer.call_count == 1
        call_args = MockServer.call_args[1]
        # new_swarm is popped
        assert call_args["initial_peers"] == [join_code]


def test_home_worker_missing_join():
    """Verify that home-worker mode fails without --join."""
    with patch("sys.argv", [
        "petals-server",
        "--mode", "home-worker",
        "bigscience/bloom-560m",
        "--device", "cpu"
    ]), \
    patch("petals.cli.run_server.Server") as MockServer, \
    patch("sys.stderr.write") as mock_stderr:

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
