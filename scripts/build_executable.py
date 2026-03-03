import os
import sys

try:
    import PyInstaller.__main__
except ImportError:
    print("[ERROR] PyInstaller is not installed. Please install it using `pip install pyinstaller`.")
    sys.exit(1)

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'petals', 'cli', 'run_server.py')

    # We want to build src/petals/cli/run_server.py as a standalone binary
    # We name it 'petals-server'
    args = [
        script_path,
        '--name=petals-server',
        '--onefile',
        '--clean',
        '--hidden-import=petals',
        '--hidden-import=petals.cli',
    ]

    print(f"Running PyInstaller with args: {args}")
    PyInstaller.__main__.run(args)
    print("Build complete. The binary is located in the 'dist/' directory.")
