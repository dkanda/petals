import sys
from unittest.mock import MagicMock, patch
import pytest
from petals.cli.run_server import main
from petals.constants import PUBLIC_INITIAL_PEERS

@pytest.fixture
def mock_server_class():
    with patch("petals.cli.run_server.Server") as MockServer:
        yield MockServer

@pytest.fixture
def mock_dht_instance(mock_server_class):
    server_instance = MagicMock()
    mock_server_class.return_value = server_instance
    # Mock DHT visible maddrs for coordinator test
    # The server.dht attribute is accessed, so we need to mock it.
    dht_mock = MagicMock()
    dht_mock.get_visible_maddrs.return_value = ["/ip4/127.0.0.1/tcp/12345/p2p/QmHash"]
    server_instance.dht = dht_mock
    return server_instance

@pytest.fixture
def base_args():
    return [
        "run_server.py",
        "--converted_model_name_or_path", "test-model",
        "--torch_dtype", "float32",
        "--no_auto_relay",
    ]

def test_home_coordinator_mode(mock_server_class, mock_dht_instance, base_args, capsys):
    with patch.object(sys, "argv", base_args + ["--mode", "home-coordinator"]):
        with patch("petals.cli.run_server.validate_version"), \
             patch("petals.cli.run_server.parse_size"), \
             patch("torch.backends.openmp.is_available", return_value=True), \
             patch("zeroconf.Zeroconf") as mock_zeroconf, \
             patch("zeroconf.ServiceInfo") as mock_service_info:

            main()

            # Ensure Zeroconf was used to register the service
            mock_zeroconf.return_value.register_service.assert_called_once()
            mock_zeroconf.return_value.unregister_service.assert_called_once()
            mock_zeroconf.return_value.close.assert_called_once()

    # Verify Server initialized with empty initial_peers (new swarm)
    # call_args.kwargs should contain initial_peers=[]
    _, kwargs = mock_server_class.call_args
    assert kwargs.get("initial_peers") == []

    # Verify join code was printed
    captured = capsys.readouterr()
    assert "Home Swarm Coordinator Started!" in captured.out
    assert "--mode home-worker --join /ip4/127.0.0.1/tcp/12345/p2p/QmHash" in captured.out

def test_home_worker_mode(mock_server_class, mock_dht_instance, base_args):
    join_code = "/ip4/192.168.1.100/tcp/12345/p2p/QmCoordinatorHash"

    with patch.object(sys, "argv", base_args + ["--mode", "home-worker", "--join", join_code]):
        with patch("petals.cli.run_server.validate_version"), \
             patch("petals.cli.run_server.parse_size"), \
             patch("torch.backends.openmp.is_available", return_value=True):

            main()

    # Verify Server initialized with correct initial_peers
    _, kwargs = mock_server_class.call_args
    assert kwargs.get("initial_peers") == [join_code]

def test_home_worker_missing_join(mock_server_class, base_args, capsys):
    # Test fallback to auto-discovery
    with patch.object(sys, "argv", base_args + ["--mode", "home-worker"]):
        with patch("petals.cli.run_server.validate_version"), \
             patch("petals.cli.run_server.parse_size"), \
             patch("torch.backends.openmp.is_available", return_value=True), \
             patch("zeroconf.Zeroconf") as mock_zeroconf, \
             patch("zeroconf.ServiceBrowser") as mock_service_browser:

            # Simulate the browser finding the service
            def mock_browser_init(zc, type_, listener):
                mock_info = MagicMock()
                mock_info.properties = {b'join_code': b'/ip4/127.0.0.1/tcp/12345/p2p/QmHashDiscovered'}
                zc.get_service_info.return_value = mock_info
                # Directly call the listener's add_service since ServiceBrowser does it in a separate thread usually
                listener.add_service(zc, type_, "PetalsSwarm._petals._tcp.local.")

            mock_service_browser.side_effect = mock_browser_init

            main()

    # Verify Server initialized with the discovered initial_peers
    _, kwargs = mock_server_class.call_args
    assert kwargs.get("initial_peers") == ["/ip4/127.0.0.1/tcp/12345/p2p/QmHashDiscovered"]

def test_home_worker_missing_join_and_discovery_fails(mock_server_class, base_args, capsys):
    # Test error when auto-discovery fails
    with patch.object(sys, "argv", base_args + ["--mode", "home-worker"]):
        with patch("petals.cli.run_server.validate_version"), \
             patch("petals.cli.run_server.parse_size"), \
             patch("torch.backends.openmp.is_available", return_value=True), \
             patch("zeroconf.Zeroconf") as mock_zeroconf, \
             patch("zeroconf.ServiceBrowser"), \
             patch("time.sleep"):

            with pytest.raises(SystemExit) as excinfo:
                main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Could not automatically discover a coordinator. --join code is required for --mode home-worker" in captured.err

def test_default_mode(mock_server_class, mock_dht_instance, base_args):
    with patch.object(sys, "argv", base_args):
        with patch("petals.cli.run_server.validate_version"), \
             patch("petals.cli.run_server.parse_size"), \
             patch("torch.backends.openmp.is_available", return_value=True):

            main()

    # Verify default initial peers
    _, kwargs = mock_server_class.call_args
    assert kwargs.get("initial_peers") == PUBLIC_INITIAL_PEERS
