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
        
        for item in self.data:
            text = item["text"]
            source = item["source"]
            
            # --- Strategy: Heuristic Section Splitting ---
            # We will try to identify common resume sections and create specific QA pairs for them.
            # This allows the model to answer "What is his education?" directly.
            
            # 1. Full Resume (The "Master" Context)
            # We still keep this so it can answer "Tell me everything" or "Who is he?"
            full_text = text[:3500]
            processed_data.append({
                "text": f"<|user|>\nWho is Jiggy Kakkad and what is his full profile?\n<|end|>\n<|assistant|>\n{full_text}\n<|end|>",
                "source": source
            })
            
            # 2. Contact Details (Usually at the start)
            # Heuristic: Take the first 300 characters or until the first section header
            headers = ["SUMMARY", "CORE SKILLS", "EXPERIENCE", "EDUCATION", "SKILLS"]
            first_header_idx = len(text)
            for header in headers:
                idx = text.find(header)
                if idx != -1 and idx < first_header_idx:
                    first_header_idx = idx
            
            contact_info = text[:first_header_idx].strip()
            if contact_info:
                processed_data.append({
                    "text": f"<|user|>\nWhat are Jiggy Kakkad's contact details and location?\n<|end|>\n<|assistant|>\n{contact_info}\n<|end|>",
                    "source": source
                })
                # Add a specific one for "number" or "email" if they exist in the text
                if "04" in contact_info: # Simple check for Aus mobile
                     processed_data.append({
                        "text": f"<|user|>\nWhat is Jiggy Kakkad's phone number?\n<|end|>\n<|assistant|>\n{contact_info}\n<|end|>",
                        "source": source
                    })

            # 3. Section Extraction
            # We split the text by known headers
            lower_text = text.lower()
            
            sections_map = {
                "education": ["education", "academic background"],
                "experience": ["experience", "employment history", "work history"],
                "skills": ["skills", "core skills", "technologies", "technical skills"],
                "summary": ["summary", "profile", "objective"]
            }
            
            for section_name, keywords in sections_map.items():
                start_idx = -1
                used_keyword = ""
                
                # Find the start of the section
                for kw in keywords:
                    idx = lower_text.find(kw)
                    if idx != -1:
                        start_idx = idx
                        used_keyword = kw
                        break
                
                if start_idx != -1:
                    # Find the end of this section (start of the next section)
                    end_idx = len(text)
                    
                    # Search for the nearest NEXT header
                    for other_section, other_keywords in sections_map.items():
                        if other_section == section_name: continue
                        
                        for other_kw in other_keywords:
                            other_idx = lower_text.find(other_kw, start_idx + len(used_keyword))
                            if other_idx != -1 and other_idx < end_idx:
                                end_idx = other_idx
                    
                    # Extract content
                    # We include the header in the content so the model sees "EDUCATION..."
                    section_content = text[start_idx:end_idx].strip()
                    
                    if len(section_content) > 20: # Ignore empty/tiny sections
                        # Generate QA Pair
                        question = f"What is Jiggy Kakkad's {section_name}?"
                        processed_data.append({
                            "text": f"<|user|>\n{question}\n<|end|>\n<|assistant|>\n{section_content}\n<|end|>",
                            "source": source
                        })
                        
                        # Add a variation
                        processed_data.append({
                            "text": f"<|user|>\nTell me about his {section_name}.\n<|end|>\n<|assistant|>\n{section_content}\n<|end|>",
                            "source": source
                        })

        logger.info(f"Created {len(processed_data)} targeted training samples from {len(self.data)} documents")
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
