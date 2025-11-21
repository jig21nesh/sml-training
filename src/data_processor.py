import os
import glob
import json
import logging
from typing import List, Dict
from pypdf import PdfReader
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFDataProcessor:
    """
    A class to process PDF files and prepare them for fine-tuning.
    """
    def __init__(self, pdf_dir: str, processed_dir: str):
        """
        Initialize the PDFDataProcessor.

        Args:
            pdf_dir (str): Directory containing PDF files.
            processed_dir (str): Directory to save processed data.
        """
        self.pdf_dir = pdf_dir
        self.processed_dir = processed_dir
        self.data: List[Dict[str, str]] = []
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def load_pdfs(self) -> None:
        """
        Reads all PDFs from the specified directory and extracts text.
        """
        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")

        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file}...")
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                if text.strip():
                    self.data.append({"text": text, "source": os.path.basename(pdf_file)})
                    logger.info(f"Successfully extracted text from {pdf_file}")
                else:
                    logger.warning(f"No text found in {pdf_file}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")

    def process_text(self, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, str]]:
        """
        Chunks the extracted text into smaller segments for training.
        
        Args:
            chunk_size (int): Maximum number of characters per chunk.
            overlap (int): Number of characters to overlap between chunks.
            
        Returns:
            List[Dict[str, str]]: List of chunks with source metadata.
        """
        processed_data = []
        
        processed_data = []
        
        for item in self.data:
            text = item["text"]
            source = item["source"]
            
            # For resumes/small docs, we want the WHOLE text in one go if possible.
            # Phi-3 has 4k context. A 2-page resume fits easily.
            # We'll just take the first 3500 characters to be safe and avoid cutting off.
            
            full_text = text[:3500] 
            
            formatted_text = f"<|user|>\nWho is Jiggy Kakkad and what are his details?\n<|end|>\n<|assistant|>\n{full_text}\n<|end|>"
            
            processed_data.append({
                "text": formatted_text,
                "source": source
            })
        
        logger.info(f"Created {len(processed_data)} training samples (one per document)")
        return processed_data

    def prepare_dataset(self, output_filename: str = "train.jsonl") -> None:
        """
        Saves the processed data as a JSONL file and creates a Hugging Face Dataset.
        
        Args:
            output_filename (str): Name of the output file.
        """
        if not self.data:
            logger.warning("No data loaded. Call load_pdfs() first.")
            return

        chunks = self.process_text()
        output_path = os.path.join(self.processed_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + '\n')
            
            logger.info(f"Saved processed dataset to {output_path}")
            
            # Verify we can load it as a HF dataset
            dataset = Dataset.from_json(output_path)
            logger.info(f"Successfully created Hugging Face dataset with {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")

if __name__ == "__main__":
    # Example usage
    processor = PDFDataProcessor(
        pdf_dir="data/pdfs",
        processed_dir="data/processed"
    )
    processor.load_pdfs()
    processor.prepare_dataset()
