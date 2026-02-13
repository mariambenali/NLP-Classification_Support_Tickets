from sentence_transformers import SentenceTransformer




MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

def preprocess_embeddings(texts):
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

