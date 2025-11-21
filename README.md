# 🚀 Phi-3 PDF Fine-Tuner

Hi there! Welcome to your personal AI training studio.

This project helps you take a bunch of your own PDF documents (like resumes, reports, or manuals) and teach a smart little AI model (Phi-3) all about them. It's like giving the AI a crash course on your specific knowledge base!

## What's Inside?

*   **Data Processor**: Reads your PDFs and turns them into bite-sized chunks the AI can learn from.
*   **Fine-Tuner**: The brain of the operation. It uses efficient training (QLoRA) to update the model without needing a supercomputer.
*   **Inference**: A chat interface where you can talk to your newly trained AI.
*   **Evaluator**: A tool to check how "confused" the model is (perplexity score).

## Getting Started

### 1. Prerequisites
You'll need Python installed. If you're on a Mac with Apple Silicon (M1/M2/M3), great news! We've optimized the scripts to run smoothly on your hardware.

### 2. Installation
First, grab the dependencies. We recommend using a virtual environment.

```bash
# If you use uv (recommended)
uv sync

# Or standard pip
pip install -r requirements.txt
```

## How to Train Your Dragon (Model) 🐉

### Step 1: Feed the Beast
Create a folder called `data/pdfs` and drop all your PDF files in there. The more relevant text, the better!

### Step 2: Prep the Food
Run the processor to convert those PDFs into a training format:
```bash
python src/data_processor.py
```
You'll see a new file pop up at `data/processed/train.jsonl`.

### Step 3: The Training Montage
Now for the magic. Run the fine-tuning script:
```bash
python src/fine_tuner.py
```
*Note: If you have a small dataset (like just one resume), the script is tuned to loop over it many times so the model actually learns.*

### Step 4: Chat Time!
Once training is done, talk to your model:
```bash
python src/inference.py
```
Ask it questions about the documents you provided and see what it knows!

## A Note for Mac Users 🍎
We've handled some tricky compatibility issues for you:
*   **MPS Support**: The scripts automatically detect if you're on a Mac and switch to settings that play nice with Apple's metal performance shaders.
*   **Stability**: We use `float32` precision on Macs to prevent those pesky "NaN" errors that can happen during training.

Happy training!
