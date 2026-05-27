import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading AI Tutor...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("AI Tutor Ready! Type quit to exit")

while True:
    q = input("\nYou: ")
    if q.lower() == "quit":
        break
    prompt = f"### Instruction:\n{q}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1].strip()
    print(f"Tutor: {response}")
