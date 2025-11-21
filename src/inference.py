import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTester:
    """
    A class to test the fine-tuned model via inference.
    """
    def __init__(self, base_model_name: str, adapter_path: str):
        """
        Initialize the ModelTester.

        Args:
            base_model_name (str): The base model ID.
            adapter_path (str): Path to the fine-tuned adapter.
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """
        Loads the base model and merges the adapter.
        """
        if self.model is not None:
            return

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

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generates a response for a given prompt.
        
        Args:
            prompt (str): The input text.
            max_new_tokens (int): Maximum tokens to generate.
            
        Returns:
            str: The generated response.
        """
        if not self.model:
            self.load_model()
            
        # Format prompt for Phi-3 Instruct if needed, or just raw
        # Using a simple template for now
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant part if possible, or return whole thing
        try:
            response = response.split("<|assistant|>")[1].strip()
        except IndexError:
            pass
            
        return response

    def interactive_mode(self):
        """
        Starts an interactive CLI session with history.
        """
        print("Starting interactive mode. Type 'exit' to quit.")
        print("Tip: Start by asking 'Who is Jiggy Kakkad and what are his details?' to load the context.")
        
        if not self.model:
            self.load_model()

        history = []
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Construct prompt with history
            full_prompt = ""
            for turn in history:
                full_prompt += f"<|user|>\n{turn['user']}<|end|>\n<|assistant|>\n{turn['assistant']}<|end|>\n"
            
            full_prompt += f"<|user|>\n{user_input}<|end|>\n<|assistant|>"
            
            # Tokenize and generate
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=500, # Increased token limit for full resume
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode only the new part
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # DEBUG: Print raw output to see what's happening
            # print(f"DEBUG RAW: {generated_text}")
            
            # Extract the last assistant response
            try:
                # Split by assistant tag and take the last part
                parts = generated_text.split("<|assistant|>")
                if len(parts) > 1:
                    response = parts[-1]
                    # Remove <|end|> if present
                    response = response.split("<|end|>")[0].strip()
                else:
                    # Fallback: try to remove the prompt manually if tags aren't found
                    response = generated_text.replace(full_prompt, "").strip()
            except Exception as e:
                response = generated_text # Fallback to everything

            
            print(f"Phi3: {response}")
            
            # Update history
            history.append({"user": user_input, "assistant": response})
            
            # Keep history manageable (last 3 turns)
            if len(history) > 3:
                history.pop(0)

if __name__ == "__main__":
    # Example usage
    tester = ModelTester("microsoft/Phi-3-mini-4k-instruct", "models/phi3-finetuned")
    tester.interactive_mode()
