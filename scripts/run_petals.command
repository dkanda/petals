#!/bin/bash

echo "=========================================="
echo "Petals One-Click Runner (macOS/Linux)"
echo "=========================================="

# Ensure script is run from the directory containing it, or fallback to current dir
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Helper to execute
execute_from() {
    # Move to the project root if the script is in "scripts/" folder
    if [[ "$SCRIPT_DIR" == *"scripts" ]]; then
        cd "$SCRIPT_DIR/.."
    fi
}

execute_from

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    echo "Please install Python 3.8+ (e.g., via Homebrew on macOS or apt on Linux)."
    exit 1
fi

VENV_DIR="petals_venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "[INFO] Upgrading pip..."
python3 -m pip install --upgrade pip > /dev/null 2>&1

# Install Petals
if [ -f "setup.cfg" ]; then
    echo "[INFO] Installing Petals from local repository..."
    python3 -m pip install -e .
else
    echo "[INFO] Installing Petals from git..."
    python3 -m pip install git+https://github.com/bigscience-workshop/petals
fi

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Petals dependencies."
    exit 1
fi

echo "[INFO] Setup complete!"

# Start the server with arguments
echo "[INFO] Starting Petals server..."
python3 -m petals.cli.run_server "$@"

# If launched via double-click on macOS, keep the terminal open
if [ "$TERM_PROGRAM" == "Apple_Terminal" ] || [ "$TERM_PROGRAM" == "iTerm.app" ]; then
    echo ""
    echo "Press Enter to close this window..."
    read -r
fi
