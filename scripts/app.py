import streamlit as st
import pandas as pd
import os
import pickle
from dotenv import load_dotenv
from helper_functions import log_info, log_error
from transformers import pipeline

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, os.getenv('DATA_DIR'), "output")

# Ensure output directory exists
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# Load summarization pipeline
@st.cache_resource
def load_summarization_pipeline():
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        return summarizer
    except Exception as e:
        log_error(f"Error loading summarizer: {e}")
        st.error("Failed to load summarization model.")
        return None

summarizer = load_summarization_pipeline()

# Streamlit UI
st.title("📝 IMDB Review Summarization App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Summarization", "Batch Summarization"])

if page == "Single Summarization":
    st.header("Summarize a Review")
    input_text = st.text_area("Enter the movie review to summarize:")

    if st.button("Summarize"):
        if summarizer and input_text.strip():
            try:
                summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                st.success("Summary:")
                st.write(summary)
                log_info("Summarization successful")
            except Exception as e:
                log_error(f"Summarization error: {e}")
                st.error("Error during summarization.")
        else:
            st.warning("Please enter a review to summarize.")

elif page == "Batch Summarization":
    st.header("Batch Review Summarization")
    uploaded_file = st.file_uploader("Upload a CSV file containing IMDB reviews", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV must contain a 'review' column.")
            else:
                st.info("Summarizing reviews. Please wait...")
                df['summary'] = df['review'].apply(
                    lambda x: summarizer(x, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                )

                output_path = os.path.join(DATA_OUTPUT_DIR, "summarized_reviews.csv")
                df.to_csv(output_path, index=False)

                st.success("Batch summarization complete!")
                st.write(df[['review', 'summary']])
                log_info("Batch summarization completed.")
        except Exception as e:
            log_error(f"Batch summarization error: {e}")
            st.error("An error occurred during batch summarization.")
