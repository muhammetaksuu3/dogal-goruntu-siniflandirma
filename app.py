import streamlit as st
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from model import SimpleCNN
from utils.preprocessing import get_transforms
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# Sınıf isimleri (Intel Image Classification sınıfları)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Modeli yükle
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Görüntüyü sınıflandır
def predict_with_probs(image, model, device, class_names):
    transform = get_transforms()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
    return predicted_idx, probabilities.cpu().numpy()


# Performans değerlendirme fonksiyonu (test verisi üzerinden)
def evaluate_model_streamlit(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    return acc, precision, recall, report


# === Streamlit Arayüzü ===
st.set_page_config(page_title="Intel Görüntü Sınıflandırıcı", layout="centered")
st.title("🌍 Intel Görüntü Sınıflandırıcı")
st.write("Bir doğal ortam fotoğrafı yükleyin, sistem sınıfını tahmin etsin.")

# Bilgilendirme kutusu — sınıfları göster
st.info(
    "📂 Modelin tahmin edebileceği sınıflar:\n"
    "- buildings 🏢\n"
    "- forest 🌲\n"
    "- glacier ❄️\n"
    "- mountain ⛰️\n"
    "- sea 🌊\n"
    "- street 🛣️"
)

# Görüntü yükleme alanı
uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

# Modeli önceden yükle
model, device = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    if st.button("Tahmin Et"):
        predicted_idx, probabilities = predict_with_probs(image, model, device, CLASS_NAMES)
        predicted_class = CLASS_NAMES[predicted_idx]

        st.success(f"Tahmin Edilen Sınıf: **{predicted_class}**")

        st.subheader("📊 Sınıf Olasılıkları")
        for i, prob in enumerate(probabilities):
            st.write(f"{CLASS_NAMES[i]}: {prob * 100:.2f}%")

# Performans raporunu gösteren buton
if st.button("📊 Model Performansını Göster"):
    transform = get_transforms()
    test_data = datasets.ImageFolder(root='dataset/seg_test/seg_test', transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    acc, precision, recall, report = evaluate_model_streamlit(model, test_loader, device, CLASS_NAMES)

    st.markdown("### 📈 Performans Metrikleri")
    st.write(f"**Accuracy:** `{acc:.4f}`")
    st.write(f"**Precision:** `{precision:.4f}`")
    st.write(f"**Recall:** `{recall:.4f}`")

    st.markdown("### 📋 Sınıf Bazlı Rapor")
    st.code(report)
