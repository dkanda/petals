import subprocess

files_to_format = [
    "src/petals/models/deepseek/config.py",
    "src/petals/models/deepseek/configuration_deepseek.py",
    "src/petals/models/deepseek/model.py",
    "src/petals/models/deepseek/modeling_deepseek.py",
    "test_deepseek.py",
    "test_deepseek_simple.py",
]

for f in files_to_format:
    subprocess.run(["python3", "-m", "black", f])
