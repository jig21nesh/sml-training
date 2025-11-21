import os
import logging
import torch
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class to evaluate the quality of the fine-tuned model.
    """
    def __init__(self, base_model_name: str, adapter_path: str):
        """
        Initialize the ModelEvaluator.

        Args:
            base_model_name (str): The base model ID.
            adapter_path (str): Path to the fine-tuned adapter.
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """
        Loads the base model and merges the adapter.
        """
        logger.info("Loading base model and adapter...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def calculate_perplexity(self, dataset_path: str, max_samples: int = 100) -> float:
        """
        Calculates perplexity on a dataset.
        
        Args:
            dataset_path (str): Path to the evaluation dataset (JSONL).
            max_samples (int): Limit number of samples for speed.
            
        Returns:
            float: The perplexity score.
        """
        if not self.model:
            self.load_model()
            
        logger.info(f"Calculating perplexity on {dataset_path}...")
        dataset = load_dataset("json", data_files=dataset_path, split="train") # Using train split as generic loader
        
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
            
        encodings = self.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
        max_length = self.model.config.max_position_embeddings
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = math.exp(torch.stack(nlls).mean())
        logger.info(f"Perplexity: {ppl}")
        return ppl

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator("microsoft/Phi-3-mini-4k-instruct", "models/phi3-finetuned")
    evaluator.calculate_perplexity("data/processed/train.jsonl")
