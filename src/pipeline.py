import pandas as pd
from preprocessing import preprocess_dataframe, preprocess_embeddings
from training import load_model, predict_model
import os


'''
Charger données
Appeler preprocessing
Charger modèle
Faire prédictions
Sauvegarder résultats
Logguer métriques
'''

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    # load_model
    model = load_model(os.path.join(BASE_DIR, "models", "rfc_ticket_model.pkl"))

    # predictions
    prediction = predict_model(model, ["text"])
    df["prediction"] = prediction

    # save_model
    df.to_csv(os.path.join(BASE_DIR, "data", "predictions.csv"), index=False)

    print("Pipeline terminé avec succès.")


if __name__ == "__main__":
    pipeline_nlp()