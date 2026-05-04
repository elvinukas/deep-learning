from __future__ import annotations

import os

import gradio as gr
from PIL import Image

from model_utils import CLASSES, DEFAULT_THRESHOLD, load_model, predict_image


CHECKPOINT_PATH = os.getenv("MODEL_CKPT", "./model.pth")
model, device = load_model(CHECKPOINT_PATH or None)

TARGET_LABELS = ["Tree", "Pizza", "Tomato"]


def _status_card(label: str, score: float, active: bool) -> str:
    bg = "#dcfce7" if active else "#f3f4f6"
    border = "#22c55e" if active else "#d1d5db"
    text = "#166534" if active else "#374151"
    state = "Present" if active else "Not detected"
    return (
        f"<div style='padding:14px;border-radius:12px;border:2px solid {border};"
        f"background:{bg};margin-bottom:10px;'>"
        f"<div style='font-weight:700;font-size:16px;color:{text};'>{label}</div>"
        f"<div style='font-size:13px;color:{text};'>{state}</div>"
        f"<div style='font-size:12px;color:{text};opacity:0.9;'>score: {score:.3f}</div>"
        f"</div>"
    )


def predict(uploaded_image: Image.Image, threshold: float = DEFAULT_THRESHOLD):
    if uploaded_image is None:
        return "".join(_status_card(lbl, 0.0, False) for lbl in TARGET_LABELS), []

    results, probs = predict_image(uploaded_image, model, device, threshold=threshold)

    score_map = {}
    for item in results:
        score_map[item["label"]] = float(item["score"])

    # Fill in scores for labels not above threshold so UI can show all classes.
    class_to_prob = {label: float(prob) for label, prob in zip(CLASSES, probs)}
    for label in TARGET_LABELS:
        if label not in score_map and label in class_to_prob:
            score_map[label] = class_to_prob[label]

    cards = []
    for label in TARGET_LABELS:
        score = score_map.get(label, 0.0)
        cards.append(_status_card(label, score, score >= threshold))

    return "".join(cards), results


with gr.Blocks() as demo:
    gr.Markdown("# Multi-label Image predictor")
    gr.Markdown("Upload an image. Labels light up green when detected.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Image")
        with gr.Column():
            threshold_input = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01, label="Threshold")
            status_html = gr.HTML(label="Detected classes")
            output_json = gr.JSON(label="Raw predictions")
            run_button = gr.Button("Predict")

    run_button.click(predict, inputs=[image_input, threshold_input], outputs=[status_html, output_json])
    image_input.change(predict, inputs=[image_input, threshold_input], outputs=[status_html, output_json])




if __name__ == "__main__":
    demo.launch()
