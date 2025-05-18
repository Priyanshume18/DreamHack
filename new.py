import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from together import Together

st.set_page_config(page_title="Skin Disease Detector & Health Chatbot", layout="wide")  # MUST BE FIRST

# ================== Load Hugging Face Model ================== #
@st.cache_resource
def load_hf_model():
    repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
    processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)
    model.eval()
    return model, processor

hf_model, image_processor = load_hf_model()

# Class names
class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans',
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus',
    'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 'Molluscum Contagiosum',
    'Mycosis Fungoides', 'Neurofibromatosis', 'Papilomatosis Confluentes And Reticulate',
    'Pediculosis Capitis', 'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis',
    'Tinea Corporis', 'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma',
    'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma',
    'vascular lesion'
]

# ================== Layout ================== #
col1, col2 = st.columns([7, 3])

# ============== LEFT COLUMN: Skin Disease Detector ============== #
with col1:
    st.title("🧑‍⚕️ Skin Disease Detector")
    uploaded_file = st.file_uploader("Upload a clear skin image:", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            encoding = image_processor(image.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                outputs = hf_model(**encoding)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

            predicted_class_name = class_names[predicted_class_idx]

            st.subheader("🔍 Prediction:")
            st.success(f"**Detected Skin Condition:** {predicted_class_name}")

# ============== RIGHT COLUMN: Medical Chatbot ============== #
with col2:
    st.markdown("### 🩺 Health Chatbot")

    # Initialize Together API
    client = Together(api_key="YOUR API KEY") 

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a powerful medical assistant. Respond to hi, hello questions with activeness. "
                    "As soon as someone gives you any symptoms you predict all possible diseases and ask questions to "
                    "narrow down to exact disease. Keep the answers as concise as possible."
                )
            }
        ]

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type your symptoms or say hello...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            full_response = ""
            response_area = st.empty()

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=st.session_state.messages,
                stream=True
            )

            for token in response:
                if hasattr(token, 'choices') and token.choices[0].delta:
                    content = token.choices[0].delta.content
                    if content:
                        full_response += content
                        response_area.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
