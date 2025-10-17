import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "HuggingFaceH4/zephyr-7b-alpha"

# 1. Define the 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Instantiate the pipeline
text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer
)

# 4. Run inference
prompt = "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate.</s>\n<|user|>\nWhat is quantization?</s>\n<|assistant|>\n"
result = text_generation_pipeline(prompt, max_new_tokens=100)
print(result['generated_text'])