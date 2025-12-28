import torch
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

# DeepSeek V3 model ID
model_name = "deepseek-ai/DeepSeek-V3"

print(f"Loading model: {model_name}")

try:
    # Now that we updated DistributedDeepseekConfig, we can just load it normally!
    # trust_remote_code=True is needed for DeepSeek V3 code from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    print("Model loaded successfully (connected to swarm).")

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0]))

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"An error occurred: {e}")
