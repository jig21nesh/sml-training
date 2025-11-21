import os
import logging
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fine_tuner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phi3FineTuner:
    """
    A class to fine-tune the Phi3 model using QLoRA.
    """
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", output_dir: str = "models/phi3-finetuned"):
        """
        Initialize the Phi3FineTuner.

        Args:
            model_name (str): The Hugging Face model ID.
            output_dir (str): Directory to save the fine-tuned model.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
        # Check for GPU availability
        self.device_map = "auto"
        if torch.backends.mps.is_available():
            self.device_map = "mps"
            logger.info("Using MPS (Apple Silicon) device.")
        elif torch.cuda.is_available():
            self.device_map = "cuda"
            logger.info("Using CUDA device.")
        else:
            logger.warning("No GPU detected. Training will be extremely slow on CPU.")
            self.device_map = "cpu"

    def setup_model(self):
        """
        Loads the model and tokenizer with 4-bit quantization configuration.
        """
        logger.info(f"Loading model {self.model_name}...")
        
        # QLoRA configuration
        bnb_config = None
        if self.device_map != "mps":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            # Note: MPS does not support 4-bit quantization via bitsandbytes natively in all versions yet.
            # If on MPS, we might need to fallback to standard loading or 8-bit if supported.
            # For this script, we'll try standard 4-bit load, but catch errors for MPS fallback if needed.
            
            if self.device_map == "mps":
                logger.warning("MPS detected. Using bfloat16 for M-series performance.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                self.model = prepare_model_for_kbit_training(self.model)

            logger.info("Model and tokenizer loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def setup_lora(self):
        """
        Configures LoRA adapters for the model.
        """
        logger.info("Setting up LoRA configuration...")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear" # Target all linear layers for better performance
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(self, dataset_path: str, epochs: int = 50, batch_size: int = 1, learning_rate: float = 2e-4):
        """
        Runs the training loop.

        Args:
            dataset_path (str): Path to the processed JSONL dataset.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size per device.
            learning_rate (float): Learning rate.
        """
        if not self.model:
            self.setup_model()
            if self.device_map != "mps": # Skip prepare_kbit for MPS as we loaded in fp16
                 self.setup_lora()
            else:
                 # Still use LoRA for MPS but on fp16 model
                 peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules="all-linear"
                )
                 self.model = get_peft_model(self.model, peft_config)
                 self.model.print_trainable_parameters()

        logger.info(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            logging_steps=1,
            num_train_epochs=epochs,
            save_steps=500,
            fp16=True if self.device_map == "cuda" else False, # MPS uses float32/16 but fp16 arg is for CUDA amp
            # bf16=True, # Enable if GPU supports it (Ampere+)
            optim="paged_adamw_8bit" if self.device_map == "cuda" else "adamw_torch",
            report_to="none" # Disable wandb for simplicity
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        logger.info("Starting training...")
        trainer.train()
        
        logger.info(f"Saving model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Training complete.")

if __name__ == "__main__":
    # Example usage
    fine_tuner = Phi3FineTuner()
    fine_tuner.train("data/processed/train.jsonl")
