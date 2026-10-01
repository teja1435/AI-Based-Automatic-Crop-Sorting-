"""Streamlit interface for the AI-based crop sorting prototype."""

import json
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.crop_sorter.model import load_model, predict_image

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "crop_sorter.joblib"
RULES_PATH = ROOT / "config" / "sorting_rules.json"

st.set_page_config(page_title="AI Crop Sorter", page_icon="🌱", layout="centered")
st.title("🌱 AI-Based Automatic Crop Sorting")
st.write("Upload a crop photo to classify its quality category and view a suggested sorting destination.")
st.caption("Prototype only: the app provides image-based guidance and does not control sorting hardware.")

try:
    classifier = load_model(MODEL_PATH)
except FileNotFoundError as exc:
    st.info("Train a model before using image predictions.")
    st.code("python train.py --data-dir data/train\nstreamlit run app.py", language="powershell")
    st.caption(str(exc))
    st.stop()

with RULES_PATH.open(encoding="utf-8") as file:
    rules = json.load(file)

uploaded = st.file_uploader("Choose a crop image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (OSError, UnidentifiedImageError):
        st.error("This file could not be read as an image. Try a JPG or PNG file.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)
    label, score, ranked = predict_image(classifier, image)
    destinations = rules.get("destinations", {})
    destination = destinations.get(label, rules.get("default_destination", "Manual inspection"))
    st.subheader("Sorting result")
    col1, col2 = st.columns(2)
    col1.metric("Predicted category", label)
    col2.metric("Suggested destination", destination)
    st.caption(f"Relative model score: {score:.1%}. This is not a calibrated probability.")
    with st.expander("See all class scores"):
        for class_name, class_score in ranked:
            st.write(f"**{class_name}** — {class_score:.1%}")
