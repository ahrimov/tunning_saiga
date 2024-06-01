import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel

model_id = "IlyaGusev/saiga_llama3_8b"
default_system_promt = "Ты помогающий людям узнать больше о капибарах."
peft_model_id = "./outputs/tunning_model_v2/"

def create_user_promt(query):
    return {
        "role": "user",
        "content": query
    }

def create_assistant_response(response):
    return {
        "role": "assistant",
        "content": response
    }

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
generation_config = GenerationConfig.from_pretrained(model_id)

history = [{
    "role": "system",
    "content": default_system_promt
}]

while True:
    query = input("User: ")
    user_promt = create_user_promt(query)
    history.append(user_promt)
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    assistant_response = create_assistant_response(output)
    history.append(assistant_response)
    print("Assistant: ", output)
    print("\n==============================\n")