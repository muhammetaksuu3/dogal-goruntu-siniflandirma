import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FruitCNN

# Dataset yolu
data_dir = "C:/Users/muhammet/Desktop/yapay_zeka_proje/fruits-360/fruits-360_100x100/fruits-360"

# Ön işleme
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

test_dataset = ImageFolder(os.path.join(data_dir, "Test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model ve cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruitCNN(num_classes=len(test_dataset.classes)).to(device)
model.load_state_dict(torch.load("fruit_model.pt"))
model.eval()

# Test işlemi
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎯 Test doğruluğu: %{accuracy:.2f}")
