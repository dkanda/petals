import os
import stat
import pytest
import subprocess

def test_scripts_exist():
    """Verify that all runner and builder scripts exist."""
    scripts = [
        "scripts/run_petals.bat",
        "scripts/run_petals.command",
        "scripts/build_executable.py"
    ]
    for script in scripts:
        assert os.path.exists(script), f"Expected script {script} to exist."

def test_run_petals_command_executable():
    """Verify that the macOS/Linux command script is executable."""
    script_path = "scripts/run_petals.command"
    st = os.stat(script_path)
    assert bool(st.st_mode & stat.S_IXUSR), f"Script {script_path} should be executable."

def test_build_executable_imports_correctly():
    """Verify that the build script imports the necessary tools and runs in check mode (dry-run if possible, or just syntax check).
    We can just run it using python -m py_compile to ensure no syntax errors."""
    result = subprocess.run(["python", "-m", "py_compile", "scripts/build_executable.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error in build_executable.py: {result.stderr}"
