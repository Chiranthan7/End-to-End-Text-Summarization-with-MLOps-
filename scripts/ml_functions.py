import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from helper_functions import log_info, log_error

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))

# Ensure Artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_summarization_pipeline(model_name="t5-small"):
    """
    Loads a pretrained Hugging Face summarization pipeline.
    """
    try:
        log_info(f"Loading summarization model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        log_info("Summarization pipeline loaded successfully.")
        return summarizer
    except Exception as e:
        log_error(f"Failed to load summarization model: {e}")
        return None

def summarize_batch(text_list, summarizer, max_length=60, min_length=20):
    """
    Summarizes a list of texts using the given summarizer pipeline.
    """
    summaries = []
    for i, text in enumerate(text_list):
        try:
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            summaries.append(result)
        except Exception as e:
            log_error(f"Summarization failed at index {i}: {e}")
            summaries.append("ERROR")
    return summaries

def evaluate_summary_length(original_texts, summaries):
    """
    Simple evaluation: logs average lengths of original vs summary.
    """
    try:
        original_lengths = [len(text.split()) for text in original_texts]
        summary_lengths = [len(text.split()) for text in summaries if text != "ERROR"]
        avg_original = sum(original_lengths) / len(original_lengths)
        avg_summary = sum(summary_lengths) / len(summary_lengths)
        compression_ratio = avg_summary / avg_original

        log_info(f"📊 Avg. Original Length: {avg_original:.2f} words")
        log_info(f"📉 Avg. Summary Length: {avg_summary:.2f} words")
        log_info(f"🔁 Compression Ratio: {compression_ratio:.2f}")
        
        return avg_original, avg_summary, compression_ratio
    except Exception as e:
        log_error(f"Evaluation failed: {e}")
        return None, None, None
