import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from helper_functions import log_info, log_error

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))

# Ensure Artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Define output path for processed data (optional)
PROCESSED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "processed_imdb_reviews.csv")

def load_and_clean_data(file_path):
    """
    Load IMDB dataset and perform basic cleaning.
    """
    try:
        df = pd.read_csv(file_path)
        if 'review' not in df.columns:
            log_error("'review' column not found in dataset.")
            return None

        df.dropna(subset=['review'], inplace=True)
        df['review'] = df['review'].astype(str).str.strip()
        df = df[df['review'].str.len() > 20]  # Filter out very short reviews
        log_info("IMDB dataset loaded and cleaned.")
        return df

    except Exception as e:
        log_error(f"Error loading data: {e}")
        return None

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the data into train and test sets.
    """
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        log_info("Data successfully split into train and test sets.")
        return train_df, test_df
    except Exception as e:
        log_error(f"Error splitting data: {e}")
        return None, None

def save_processed_data(df, output_path=PROCESSED_DATA_PATH):
    """
    Save cleaned dataframe for later use.
    """
    try:
        df.to_csv(output_path, index=False)
        log_info(f"Processed data saved to {output_path}")
    except Exception as e:
        log_error(f"Error saving processed data: {e}")
