import joblib
import os




BASE_DIR = (
        "/Users/miriambenali/Desktop/Project-Simplon/"
        "NLP-Classification_Support_Tickets--with-MLOps"
    )

model_path=os.path.join(BASE_DIR, "models/rfc_ticket_model.pkl")

def load_model():
    model= joblib.load(model_path)

    return model


def predict_model(model, text):
    predictions= model.predict(text)

    return predictions



