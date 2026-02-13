from sentence_transformers import SentenceTransformer
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

def preprocess_embeddings(texts):
    clean_text = os.path.join(BASE_DIR,
                    "data/processed/tickets_clean.csv")

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

