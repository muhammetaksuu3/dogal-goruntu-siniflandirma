import matplotlib
matplotlib.use("TkAgg")  # Matplotlib çökmesini önlemek için backend ayarı

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

# Veri klasörünün yolu (klasör yapısına göre güncel)
data_dir = "C:/Users/muhammet/Desktop/yapay_zeka_proje/fruits-360/fruits-360_100x100/fruits-360"
# Görseller için ön işleme adımları
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Eğitim ve test veri setlerini yükle
train_dataset = ImageFolder(os.path.join(data_dir, "Training"), transform=transform)
test_dataset = ImageFolder(os.path.join(data_dir, "Test"), transform=transform)

# DataLoader ile verileri çek (şimdilik batch_size = 1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Bir örnek veri al ve görselleştir
images, labels = next(iter(train_loader))
image = images[0].permute(1, 2, 0)  # PyTorch -> Matplotlib formatı (C,H,W -> H,W,C)

# Görseli kaydet
plt.imshow(image)
plt.title(f"Etiket: {train_dataset.classes[labels[0]]}")
plt.axis("off")
plt.savefig("output.png")  # Çıktı görseli dosya olarak kaydediliyor
plt.close()

print("✅ Görsel 'output.png' olarak kaydedildi. Masaüstünden açıp görüntüleyebilirsin.")



import torch.nn as nn
import torch.nn.functional as F

class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 23 * 23, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 100x100 → 98x98 → 49x49
        x = self.pool(F.relu(self.conv2(x)))   # 47x47 → 23x23
        x = x.view(-1, 64 * 23 * 23)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x







