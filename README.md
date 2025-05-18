
````markdown
# 🧑‍⚕️ Skin Disease Detector & Health Chatbot

An intelligent healthcare application designed to make **preventative care** more **digitally accessible** through an AI-based skin disease detection system and an interactive medical chatbot—all within a single web interface.

---

## 🌐 Overview

This web-based platform integrates two core components:

### 🩺 Skin Disease Detection  
Utilizes a fine-tuned Vision Transformer model to analyze skin images and assess the presence of potential dermatological conditions. With an intuitive interface, users can upload a clear skin image and receive immediate, AI-driven feedback on skin health.

### 💬 Health Chatbot  
Powered by the LLaMA-3 large language model through the Together API, the chatbot facilitates medical conversations by:
- Responding promptly to greetings.
- Gathering symptom-related information.
- Suggesting potential conditions.
- Asking follow-up questions to refine predictions.

This solution provides **efficient, accessible, and intelligent healthcare assistance**, especially valuable for users in remote or underserved areas.

---

## 🖼️ Usability

- Upload a skin image → Receive AI-based health insights.
- Type your symptoms → Engage with a dynamic and concise medical assistant.
- Mobile-friendly, responsive, and designed for ease of use.
- The chatbot operates independently—even if the skin model file is unavailable.

---

## 🧠 Model: `inov2-base-finetuned-SkinDisease`

A fine-tuned version of **`facebook/dinov2-base`**, this model was trained on a custom skin disease dataset to classify images as either **Healthy** or **Possibly Diseased**.

### 📈 Performance:
- **Loss**: 0.1321  
- **Accuracy**: 95.57%

### 📌 Architecture Summary:
- **Base**: Vision Transformer (ViT)
- **Backbone**: Transformer Encoder (BERT-style)
- **Input Processing**: Images are split into patches, linearly embedded, and combined with positional embeddings and a `[CLS]` token.
- **Output**: Classification based on the final hidden state of the `[CLS]` token.

This approach enables strong image representation learning, ideal for downstream tasks like disease classification.

---

## 🧪 Framework Versions

| Framework       | Version   |
|-----------------|-----------|
| Transformers    | 4.33.2    |
| PyTorch         | 2.0.0     |
| Datasets        | 2.1.0     |
| Tokenizers      | 0.13.3    |

---

## ⚙️ Setup Instructions

### 🔧 Requirements

Install the required packages:
```bash
pip install streamlit torch Pillow numpy together
````

### 📂 Project Setup

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   ```

2. **Place the model file**:
   Save the `m1.pth` model checkpoint in the project root directory.

3. **Set your API key**:
   Add your Together API key in the script:

   ```python
   client = Together(api_key="YOUR_API_KEY")
   ```

4. **Run the application**:

   ```bash
   streamlit run your_script.py
   ```

> **Note**: The chatbot is functional even without the model file, ensuring continued usability.

---

## 📝 Citation

If you use this model or build upon it, please cite the original research:

```bibtex
@article{mohan2024enhancing,
  title={Enhancing skin disease classification leveraging transformer-based deep learning architectures and explainable ai},
  author={Mohan, Jayanth and Sivasubramanian, Arrun and Sowmya, V and Vinayakumar, Ravi},
  journal={arXiv preprint arXiv:2407.14757},
  year={2024}
}
```

---

## 👨‍💻 Acknowledgments

* **Model**: `dinov2-base` by Meta, fine-tuned on a specialized skin disease dataset.
* **Chatbot**: LLaMA-3 API via [Together.ai](https://www.together.ai/)
* **Interface**: Built using [Streamlit](https://streamlit.io/)

```

Let me know if you'd like me to generate badges, license sections, or screenshots to enhance the README further.
```
