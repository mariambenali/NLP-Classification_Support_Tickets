from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
embeddings_path = os.path.join(BASE_DIR, "data/processed/embeddings.npy")
labels_path = os.path.join(BASE_DIR, "data/processed/tickets_clean.csv")


def model_training():
    #load embeddings
    X = np.load(embeddings_path)

    #load label
    df= pd.read_csv(labels_path)
    y= df["type"].tolist()

    #split data
    X_train,X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    #train
    model.fit(X_train,y_train)

    #prediction
    y_pred = model.predict(X_test)

    #evaluation
    print(classification_report(y_test, y_pred))

    #save model
    model_path = os.path.join(BASE_DIR, "models/rfc_ticket_model.pkl")
    joblib.dump(model, model_path)

    return model


def load_model():
    model_path = os.path.join(BASE_DIR, "models", "rfc_ticket_model.pkl")
    return joblib.load(model_path)


def predict_model(model, embeddings):
    return model.predict(embeddings)
