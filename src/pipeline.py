import pandas as pd
from preprocessing import preprocess_dataframe
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

BASE_DIR= (
        "/Users/miriambenali/Desktop/Project-Simplon/"
        "NLP-Classification_Support_Tickets--with-MLOps"
    )

dir_path= os.path.join(BASE_DIR,
        "data/processed/tickets_clean.csv")

def pipeline_nlp():
    #load_data
    df= pd.read_csv(dir_path)

    #cleaning_data
    df= preprocess_dataframe(df, "text")

    #load_model
    model= load_model("models/rfc_ticket_model.pkl")

    #predictions
    prediction= predict_model(model,["text"])
    df["prediction"]= prediction

    #save_model
    df.to_csv("data/predictions.csv", index=False)

    print("Pipeline terminé avec succès.")


if __name__ == "__main__":
    pipeline_nlp()


