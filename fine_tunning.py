import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_id = "IlyaGusev/saiga_llama3_8b"
train_data_path = "./data/capybara_train.json"
validate_data_path = "./data/capybara_val_.json"
output_dir = "./outputs/"
output_model = output_dir + "tunning_model_v2"
max_train_steps = 50
learning_rate = 3e-4

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Квантанизация. Параметры взяты из https://huggingface.co/blog/4bit-transformers-bitsandbytes
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# Оптимизация дообучения с помощью PEFT
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# Преобработка данных
data = load_dataset( 
    "json", 
    data_files={
                'train' : train_data_path ,
                'validation' : validate_data_path
    }
)

CUTOFF_LEN = 3000

def generate_llama_prompt(data):
    return f"""<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {data['system']}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {data['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {data['bot']}<|eot_id|>"""
 
    
def tokenize(prompt):
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None
    )
    if (tokens["input_ids"][-1] != tokenizer.eos_token_id and len(tokens["input_ids"]) < CUTOFF_LEN):
        tokens["input_ids"].append(tokenizer.eos_token_id)
        tokens["attention_mask"].append(1)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
 

def generate_and_tokenize_prompt(data):
    full_prompt = generate_llama_prompt(data)
    return tokenize(full_prompt)


train_data = data["train"].map(generate_and_tokenize_prompt)
val_data = data["validation"].map(generate_and_tokenize_prompt)

# НАстройка параметров обучения модели
training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=max_train_steps,
            learning_rate=learning_rate,
            fp16=True,
            optim="adamw_torch",
            output_dir=output_dir,
            disable_tqdm=False,
            overwrite_output_dir=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# Запуск обучения модели
trainer = Trainer(
    model,
    training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)
model = torch.compile(model)
trainer.train()
model.save_pretrained(output_model)
print(f'Time: {time.time() - start_time}')
