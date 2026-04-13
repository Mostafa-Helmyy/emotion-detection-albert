import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🎭 Emotion Detection App")

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Models
# =========================


@st.cache_resource
def load_full_model():
    model = AutoModelForSequenceClassification.from_pretrained("./model_full")
    tokenizer = AutoTokenizer.from_pretrained("./model_full")
    model.to(device)
    return model, tokenizer


@st.cache_resource
def load_lora_model():
    # 🔥 الحل هنا: num_labels=6
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "albert-base-v2",
        num_labels=6
    )

    model = PeftModel.from_pretrained(base_model, "./model_lora")

    tokenizer = AutoTokenizer.from_pretrained("./model_lora")

    model.to(device)
    return model, tokenizer


# =========================
# Model Selection
# =========================
model_choice = st.selectbox(
    "Choose Model",
    ["Full Model (Best Accuracy)", "LoRA Model (Faster)"]
)

if model_choice == "Full Model (Best Accuracy)":
    model, tokenizer = load_full_model()
else:
    model, tokenizer = load_lora_model()

# =========================
# Session State
# =========================
if "text" not in st.session_state:
    st.session_state.text = ""

# =========================
# Text Input
# =========================
st.subheader("✍️ Enter your text")

user_input = st.text_area(
    "",
    value=st.session_state.text,
    height=120
)

# =========================
# Suggestions
# =========================
st.subheader("💡 Try examples")

examples = [
    "I feel amazing today!",
    "I am very sad",
    "I hate everything",
    "I am scared",
    "I love you so much",
    "This is the worst day ever",
    "I feel nervous about tomorrow",
    "Wow I didn't expect that!"
]

cols = st.columns(2)

for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex):
            st.session_state.text = ex
            st.rerun()

# =========================
# Prediction Function
# =========================


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    return emotions[pred], probs[0].tolist()


# =========================
# Predict Button
# =========================
st.markdown("---")

if st.button("🔍 Predict Emotion", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        emotion, probs = predict(user_input)

        st.success(f"🎯 Predicted Emotion: {emotion}")

        # =========================
        # Probabilities
        # =========================
        st.subheader("📊 Confidence")

        emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

        for i in range(len(emotions)):
            st.write(f"{emotions[i]}: {probs[i]:.2f}")
