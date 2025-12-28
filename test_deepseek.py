import os

import torch
from transformers import AutoConfig, AutoTokenizer

from petals import AutoDistributedModelForCausalLM
from petals.models.deepseek.config import DistributedDeepseekConfig

# DeepSeek V3 model ID
model_name = "deepseek-ai/DeepSeek-V3"

print(f"Loading model: {model_name}")

try:
    # Load config first to inspect/patch using the Distributed config class
    config = DistributedDeepseekConfig.from_pretrained(model_name, trust_remote_code=True)

    # Patch quantization config if it causes issues with this version of transformers
    # The error was "Unknown quantization type, got fp8"
    if hasattr(config, "quantization_config"):
        print(f"Original quantization config: {config.quantization_config}")
        # Try to remove it or adapt it. For Petals distributed client, we don't necessarily need the local quantization config
        # if we are just connecting to a swarm (the swarm handles the weights).
        # However, the client needs to instantiate the model class.
        # Let's try attempting to clear it or set to something valid if strictly needed.
        # But if we clear it, transformers might try to load in full precision which is fine for meta-device initialization usually used by Petals.
        del config.quantization_config  # Try removing the attribute entirely
        print("Cleared quantization config for compatibility.")

    # Connect to a distributed network hosting model layers
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    print("Model loaded successfully (connected to swarm).")
    print("Generating text...")

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0]))

except ValueError as e:
    print(f"ValueError: {e}")
except OSError as e:
    print(f"OSError: {e}")
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"An error occurred: {e}")
