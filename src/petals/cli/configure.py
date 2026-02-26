import os
import sys

import torch
import yaml
from transformers import AutoConfig

from petals.constants import PUBLIC_INITIAL_PEERS
from petals.server.block_utils import estimate_num_blocks, resolve_block_dtype
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.convert_block import QuantType

# Popular models list
POPULAR_MODELS = [
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-2-70b-chat-hf",
    "tiiuae/falcon-180B-chat",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "bigscience/bloom",
]


def run_wizard():
    print("Welcome to Petals Configuration Wizard!")
    print("This wizard will help you set up your Petals server.")
    print()

    # 1. Choose Model
    print("1. Choose a model to host:")
    for i, model in enumerate(POPULAR_MODELS):
        print(f"  {i+1}) {model}")
    print("  0) Custom model")

    while True:
        try:
            choice = input("Select a model (default: 1): ").strip()
        except KeyboardInterrupt:
            print("\nExiting wizard.")
            sys.exit(0)

        if not choice:
            model_name = POPULAR_MODELS[0]
            break
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(POPULAR_MODELS):
                model_name = POPULAR_MODELS[idx - 1]
                break
            elif idx == 0:
                model_name = input("Enter custom model name (e.g. meta-llama/Llama-2-70b-hf): ").strip()
                if model_name:
                    break
        print("Invalid selection. Please try again.")
    print(f"Selected model: {model_name}")
    print()

    # 2. Public or Private Swarm
    print("2. Swarm type:")
    print("  1) Public swarm (connect to public.petals.dev)")
    print("  2) Private swarm (start a new swarm or join a private one)")

    swarm_type = "public"
    while True:
        try:
            choice = input("Select swarm type (default: 1): ").strip()
        except KeyboardInterrupt:
            print("\nExiting wizard.")
            sys.exit(0)

        if not choice:
            swarm_type = "public"
            break
        if choice == "1":
            swarm_type = "public"
            break
        elif choice == "2":
            swarm_type = "private"
            break
        print("Invalid selection.")

    initial_peers = PUBLIC_INITIAL_PEERS if swarm_type == "public" else []
    new_swarm = False
    if swarm_type == "private":
        # Check if creating new or joining
        print("  a) Start a new private swarm")
        print("  b) Join an existing private swarm")
        while True:
            subchoice = input("Select (default: a): ").strip().lower()
            if not subchoice or subchoice == "a":
                new_swarm = True
                initial_peers = []  # logic for new swarm
                break
            elif subchoice == "b":
                peers = input("Enter one or more initial peers (comma separated): ").strip()
                if peers:
                    initial_peers = [p.strip() for p in peers.split(",")]
                    break
            print("Invalid selection.")
    print(f"Selected swarm type: {swarm_type}")
    print()

    # 3. GPU Selection
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"3. GPU Selection (Found {num_gpus} GPUs):")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  {i}) {props.name} ({props.total_memory / (1024**3):.1f} GiB)")

        print("  all) Use all GPUs (tensor parallelism)")

        selected_devices = []
        while True:
            choice = input("Select GPU(s) (default: all): ").strip().lower()
            if not choice or choice == "all":
                selected_devices = [f"cuda:{i}" for i in range(num_gpus)]
                break
            # Parsing "0,1" or "0"
            try:
                indices = [int(x) for x in choice.split(",")]
                if all(0 <= x < num_gpus for x in indices):
                    selected_devices = [f"cuda:{x}" for x in indices]
                    break
            except ValueError:
                pass
            print("Invalid selection.")
    else:
        print("3. No GPU detected. Using CPU (not recommended for production).")
        selected_devices = ["cpu"]

    print(f"Selected devices: {selected_devices}")
    print()

    # 4. Public Name
    public_name = input("4. Public name (optional, to appear on monitor): ").strip() or None
    print()

    # 5. Hugging Face Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("5. Hugging Face Token (required for gated models like Llama 2): ").strip() or None
    else:
        print("5. Hugging Face Token found in environment variables.")
    print()

    # 6. Estimate Blocks
    print("Estimating optimal number of blocks...")
    num_blocks = None
    try:
        # Load config
        block_config = AutoDistributedConfig.from_pretrained(model_name, use_auth_token=token)

        # Prepare arguments for estimation
        device = torch.device(selected_devices[0])  # Use first device for estimation logic base
        # If multiple devices, estimate handles it via tensor_parallel_devices arg

        # Determine defaults
        torch_dtype = resolve_block_dtype(block_config, "auto")
        quant_type = QuantType.NF4 if device.type == "cuda" else QuantType.NONE

        tp_devices = [torch.device(d) for d in selected_devices]

        # Default cache tokens
        is_multiquery = block_config.num_key_value_groups > 1
        attn_cache_tokens = 16384 if is_multiquery else 4096

        num_blocks = estimate_num_blocks(
            block_config=block_config,
            device=device,
            torch_dtype=torch_dtype,
            quant_type=quant_type,
            tensor_parallel_devices=tp_devices,
            attn_cache_tokens=attn_cache_tokens,
            token=token,
        )
        print(f"Estimated number of blocks: {num_blocks}")
    except Exception as e:
        print(f"Error estimating blocks: {e}")
        try:
            nb = input("Enter number of blocks manually: ").strip()
            num_blocks = int(nb) if nb else None
        except ValueError:
            num_blocks = None

    print()

    # 7. Write Config
    config_data = {
        "model": model_name,
        "public_name": public_name,
        "initial_peers": initial_peers,
        "num_blocks": num_blocks,
        "tensor_parallel_devices": selected_devices if len(selected_devices) > 1 else None,
        # "device": selected_devices[0] if len(selected_devices) == 1 else None, # run_server handles defaults
    }

    if new_swarm:
        config_data["new_swarm"] = True
        if "initial_peers" in config_data:
            del config_data["initial_peers"]

    if token:
        config_data["token"] = token

    # Remove None values
    config_data = {k: v for k, v in config_data.items() if v is not None}

    with open("config.yml", "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    print("Configuration saved to config.yml")
    print("You can now run the server with:")
    print("  python -m petals.cli.run_server")


if __name__ == "__main__":
    run_wizard()
