import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Deteksi Gambar AI vs Real", page_icon="🔍", layout="centered")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("best_effnet_ft.pth", map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🔍 Deteksi Gambar AI vs Real")
st.caption("Model: EfficientNet-B0 (Fine-Tuned) • Akurasi Test: 93,33%")

uploaded_file = st.file_uploader("Upload gambar untuk dicek", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    ai_prob = float(probs[0]) * 100
    real_prob = float(probs[1]) * 100

    st.subheader("Hasil Prediksi")
    if ai_prob >= real_prob:
        st.warning(f"**Terdeteksi: AI-Generated** ({ai_prob:.2f}%)")
    else:
        st.success(f"**Terdeteksi: Foto Asli / Real** ({real_prob:.2f}%)")

    col1, col2 = st.columns(2)
    col1.metric("Kemungkinan AI", f"{ai_prob:.2f}%")
    col2.metric("Kemungkinan Real", f"{real_prob:.2f}%")

    st.write("Persentase AI:")
    st.progress(int(round(ai_prob)))
