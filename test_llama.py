from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Choose any model available at https://health.petals.dev
model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

print(f"Loading model: {model_name}")

try:
    # Connect to a distributed network hosting model layers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

    # Run the model as if it were on your computer
    print("Generating text...")
    inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=5)
    print(tokenizer.decode(outputs[0]))
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
