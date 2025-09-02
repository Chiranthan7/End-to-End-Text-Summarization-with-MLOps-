import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"  # Fix for PyTorch module path error

import streamlit as st
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

st.set_page_config(page_title="Text Summarization", layout="wide")

# Load summarizer with device=-1 to force CPU (comment out device param if you want GPU)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)

summarizer = load_summarizer()

def summarize_texts(texts, max_samples=100):
    summaries = []
    max_input_words = 512  # HuggingFace T5-small max input approx

    for i, text in enumerate(texts[:max_samples]):
        try:
            if not isinstance(text, str) or len(text.strip()) == 0:
                summaries.append("Empty or invalid text")
                continue

            # Truncate input text by words to max_input_words
            words = text.split()
            if len(words) > max_input_words:
                text = " ".join(words[:max_input_words])

            # Fixed min and max lengths for summary output
            summary = summarizer(
                text,
                max_length=150,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error summarizing row {i}: {e}")
            summaries.append(f"ERROR: {e}")

    return summaries

st.title("📄 Text Summarization App using Hugging Face + Streamlit")

uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ Uploaded CSV must contain a column named 'text'")
        else:
            st.success(f"✅ Loaded {len(df)} records. Starting summarization...")

            with st.spinner("Generating summaries..."):
                df = df.dropna(subset=["text"]).reset_index(drop=True)
                df["summary"] = summarize_texts(df["text"].tolist(), max_samples=100)

            st.success("✅ Summarization completed!")
            st.dataframe(df[["text", "summary"]].head(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Result CSV", csv, "summarized_output.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("👆 Upload a CSV file to begin.")
