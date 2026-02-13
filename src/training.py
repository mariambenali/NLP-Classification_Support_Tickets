import joblib
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "rfc_ticket_model.pkl")


def load_model(model):
    model = joblib.load(model_path)
    return model


def predict_model(model, text):
    predictions = model.predict(text)
    return predictions