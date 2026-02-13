import pandas as pd
from preprocessing import preprocess_dataframe
from embeddings import preprocess_embeddings
from training import predict_model
import os
import joblib



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models/rfc_ticket_model.pkl")

dir_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "tickets_clean.csv"
)

def pipeline_nlp():
    # load_data
    df = pd.read_csv(dir_path)

    # cleaning_data
    df = preprocess_dataframe(df, "text")

    #embeddings
    texts = df["text"].tolist()
    embeddings = preprocess_embeddings(texts)

    #load_model
    model = joblib.load(model_path)

    # predictions
    predictions = predict_model(model, embeddings)
    df["prediction"] = predictions

    # save_model
    df.to_csv(os.path.join(
        BASE_DIR,
        "data",
        "predictions.csv"),
        index=False)

    print("Pipeline terminé avec succès.")


'''if __name__ == "__main__":
    pipeline_nlp()'''