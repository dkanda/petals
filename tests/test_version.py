from unittest.mock import MagicMock, patch

from petals.utils.version import validate_version


def test_validate_version_success():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "releases": {
                "2.2.0": {},
                "2.3.0": {},
                "2.4.0.dev1": {},  # Prerelease, should be ignored
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch("petals.utils.version.logger") as mock_logger:
            with patch("petals.__version__", "2.2.0"):
                validate_version()
                # Check if it logged that a newer version is available (2.3.0)
                assert any(
                    "A newer version 2.3.0 is available" in call.args[0] for call in mock_logger.info.call_args_list
                )


def test_validate_version_no_update():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "releases": {
                "2.2.0": {},
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch("petals.utils.version.logger") as mock_logger:
            with patch("petals.__version__", "2.2.0"):
                validate_version()
                # Should not log about a newer version
                assert not any("A newer version" in call.args[0] for call in mock_logger.info.call_args_list)


def test_validate_version_exception():
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")

        with patch("petals.utils.version.logger") as mock_logger:
            validate_version()
            # Check if it logged a warning
            mock_logger.warning.assert_called_once()
            assert "Failed to fetch the latest Petals version from PyPI:" in mock_logger.warning.call_args[0][0]
