import os
import torch
from datetime import datetime
from datasets import load_dataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import warnings

# Suppress TensorFlow CPU feature and TensorRT warnings and tokenizers parallelism warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Constants
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_DATA_PATH = 'train.jsonl'
VAL_DATA_PATH = 'validation.jsonl'
OUTPUT_DIR = "./mistral-uslaw-finetune"

def setup_tokenizer(base_model_id):
    """Initializes and returns the tokenizer with specific configurations."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", add_eos_token=True, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    """Tokenizes the dataset using the provided tokenizer."""
    def generate_and_tokenize_prompt(prompt):
        text = f"<s>[INST]{prompt['input']}[/INST] {prompt['output']}</s>"
        result = tokenizer(text, truncation=True, max_length=360, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result
    return dataset.map(generate_and_tokenize_prompt)

def setup_model_and_optimizer(base_model_id):
    """Initializes and configures the model and optimizer."""
    # Configuration for model quantization and initialization
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", quantization_config=bnb_config)

    # LoRa and PEFT configurations
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none", lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    return model

def main():
    # Setup tokenizer and datasets
    tokenizer = setup_tokenizer(BASE_MODEL_ID)
    train_dataset = tokenize_dataset(load_dataset('json', data_files=TRAIN_DATA_PATH, split='train'), tokenizer)
    val_dataset = tokenize_dataset(load_dataset('json', data_files=VAL_DATA_PATH, split='train'), tokenizer)

    # Initialize model and accelerator
    model = setup_model_and_optimizer(BASE_MODEL_ID)
    os.environ['WANDB_DISABLED'] = 'true'
    # Training configurations
    training_args =  TrainingArguments(
        output_dir=OUTPUT_DIR,
        warmup_steps=5,
        per_device_train_batch_size=4,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2.2e-5, 
        logging_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        
        save_strategy="steps",       
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training

    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()






