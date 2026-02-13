import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    file_path = os.path.join(BASE_DIR, "data", "raw", "dataset.csv")
    df = pd.read_csv(file_path)
    return df


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def cleaning_data(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)


def preprocess_dataframe(df, clean_text):
    df[clean_text] = df["text"].apply(cleaning_data)
    return df



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

def preprocess_embeddings(clean_text):
    embeddings = embedding_model.encode(
        clean_text,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings