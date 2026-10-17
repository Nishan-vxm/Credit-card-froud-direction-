import os
import joblib
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.getenv("MODEL_PATH", "credit_card_fraud_ann.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "credit_card_scaler.pkl")
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

FEATURE_NAMES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]

def load_artifacts():
    missing = [path for path in (MODEL_PATH, SCALER_PATH) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Required model artifact(s) not found: "
            + ", ".join(missing)
            + ". Place the trained model and scaler beside app.py, or set MODEL_PATH and SCALER_PATH."
        )
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

def predict_fraud(*values):
    try:
        row = np.asarray(values, dtype=np.float64).reshape(1, -1)
        if row.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} input features, received {row.shape[1]}.")

        scaled_row = scaler.transform(row)
        raw_prediction = np.asarray(model.predict(scaled_row, verbose=0)).reshape(-1)

        if raw_prediction.size != 1:
            raise ValueError(
                "This app expects a binary ANN model that returns one sigmoid probability."
            )

        probability = float(raw_prediction[0])
        if not np.isfinite(probability) or probability < 0 or probability > 1:
            raise ValueError(
                "Model output was not a probability between 0 and 1. Check the model output layer."
            )

        if probability >= THRESHOLD:
            result = "🚨 Potential fraud flagged"
        else:
            result = "✅ Likely legitimate"

        return result, f"{probability * 100:.2f}%", (
            f"Decision threshold: {THRESHOLD:.2f}. This is a model estimate, "
            "not a confirmed fraud determination."
        )
    except Exception as exc:
        return "Prediction error", "N/A", str(exc)

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    title="FraudGuard AI | Credit Card Fraud Detection",
) as demo:
    gr.Markdown(
        """
        # 🛡️ FraudGuard AI
        ### Credit Card Fraud Detection Dashboard

        Enter a transaction's 30 model features to estimate its fraud probability.
        """
    )
    with gr.Row():
        gr.Markdown("**Model:** Artificial Neural Network (ANN)")
        gr.Markdown(f"**Decision threshold:** `{THRESHOLD:.2f}`")

    with gr.Accordion("Transaction features", open=True):
        inputs = [
            gr.Number(label=feature, value=0.0, precision=6)
            for feature in FEATURE_NAMES
        ]
        gr.Markdown(
            "**Input note:** `V1`–`V28` are anonymized dataset features. "
            "Use values prepared in the same way as the model's training data. "
            "All-zero values are only useful for checking that the interface works."
        )

    analyze = gr.Button("🔍 Analyze transaction", variant="primary")
    with gr.Group():
        result_output = gr.Textbox(label="Prediction", interactive=False)
        probability_output = gr.Textbox(label="Estimated fraud probability", interactive=False)
        note_output = gr.Textbox(label="Interpretation", interactive=False)

    analyze.click(
        fn=predict_fraud,
        inputs=inputs,
        outputs=[result_output, probability_output, note_output],
    )

if __name__ == "__main__":
    demo.launch()
