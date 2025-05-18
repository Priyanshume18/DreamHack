import streamlit as st
st.set_page_config(page_title="Skin Disease Detector & Health Chatbot", layout="wide")  # MUST BE FIRST

from PIL import Image
import numpy as np
import torch
import os
from together import Together

# Safe model loading without Streamlit calls inside
@st.cache_resource
def load_model():
    try:
        model = torch.load("m1.pth", map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception:
        return None

model = load_model()

# Show a warning if model is not available
if model is None:
    st.warning("⚠️ Model file 'm1.pth' not found or failed to load. Skin disease detection will be disabled.")

# Create layout
col1, col2 = st.columns([7, 3])

# ============== LEFT COLUMN: Skin Disease Detector ============== #
with col1:
    st.title("🧑‍⚕️ Skin Disease Detector")
    uploaded_file = st.file_uploader("Upload a clear skin image:", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model:
            # Preprocess image to feed into model
            img = image.resize((224, 224))  # Resize as per model input
            img = np.array(img) / 255.0  # Normalize
            if img.ndim == 2:  # grayscale to RGB
                img = np.stack([img]*3, axis=-1)
            img = img.transpose((2, 0, 1))  # Convert to CxHxW
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

            with st.spinner("Analyzing..."):
                output = model(img_tensor)
                prediction = torch.sigmoid(output).item()

                if prediction > 0.5:
                    st.error("⚠️ Possible Skin Disease Detected!")
                else:
                    st.success("✅ Skin Appears Healthy.")
        else:
            st.info("Model not available. Please upload `m1.pth` to enable diagnosis.")

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
