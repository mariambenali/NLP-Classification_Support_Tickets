from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
import pandas as pd
from joblib import load
from sentence_transformers import SentenceTransformer
import os


def run_evidently_monitoring():
    # load data
    BASE_DIR = (
        "/Users/miriambenali/Desktop/Project-Simplon/"
        "NLP-Classification_Support_Tickets--with-MLOps"
    )
    file_path = os.path.join(
        BASE_DIR,
        "data/processed/tickets_clean.csv")
    df = pd.read_csv(file_path)

    # load model
    model_path = os.path.join(
        BASE_DIR,
        "models/rfc_ticket_model.pkl")
    model = load(model_path)

    # model embeddings
    model_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    # Split the data to simulate "past" and "present" reference_data and current_data

    reference_data = df.sample(5000, random_state=42)

    current_data = df.sample(5000, random_state=0)

    # generate predictions
    # embedding & normalize
    reference_embeddings = model_embeddings.encode(
        reference_data["clean_text"].tolist(),
        normalize_embeddings=True
    )

    current_embeddings = model_embeddings.encode(
        current_data["clean_text"].tolist(),
        normalize_embeddings=True
    )

    # prediction
    reference_data["prediction"] = model.predict(reference_embeddings)
    current_data["prediction"] = model.predict(current_embeddings)

    # define evidently mappin
    data_definition = DataDefinition(
        categorical_columns=["queue", "priority", "language", "type", "prediction"],
        classification=[
            MulticlassClassification(
                target="type",
                prediction_labels="prediction")
        ]
    )
    reference_dataset = Dataset.from_pandas(
        reference_data,
        data_definition=data_definition
    )

    current_dataset = Dataset.from_pandas(
        current_data,
        data_definition=data_definition)

    # report data drift
    drift_result = Report(metrics=[DataDriftPreset()]).run(
        reference_dataset,
        current_dataset
        )
    classification_result = Report(
        metrics=[ClassificationPreset()]).run(
            reference_dataset,
            current_dataset
        )
    # save the report
    reports_path = os.path.join(
        BASE_DIR,
        "reports")
    os.makedirs(reports_path, exist_ok=True)

    # data drift
    drift_result.save_html(
        os.path.join(
            reports_path,
            "data_drift_report.html"))

    # Classification
    classification_result.save_html(
        os.path.join(
            reports_path,
            "classification_report.html")
    )

    print("Monitoring terminé")

run_evidently_monitoring()
