# клонирование
git clone https://github.com/Jorjanoo/food-cv.git


# зависимости (CPU-версия PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib opencv-python tqdm requests "huggingface_hub[cli]" fsspec jupyterlab

# Запуск Jupyter
jupyter lab
# или
jupyter notebook

Открыть food_object_detection_pytorch.ipynb.

# 1. Imports
import os
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests

# 2. Dataset Loading
df = pd.read_csv("hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv")
df.head()

# 3. Download images + image_path
NUM_IMAGES = 200
df = df.head(NUM_IMAGES).reset_index(drop=True)

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

def download_and_get_path(row):
    url = row["image_url"]
    filename = f"{row.name}.jpg"
    filepath = images_dir / filename

    if not filepath.exists():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)

    return filename  # ВАЖНО: только имя файла

df["image_path"] = df.apply(download_and_get_path, axis=1)
df[["image_url", "image_path"]].head()

# 4. Custom Dataset
class FoodDetectionDataset(Dataset):
    def __init__(self, dataframe, image_root, transforms=None, max_samples=200):
        self.df = dataframe.head(max_samples)
        self.image_root = image_root
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        width, height = image.size
        boxes = torch.tensor([[0, 0, width, height]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # 1 = food

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target

# 5. DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

dataset = FoodDetectionDataset(df, image_root="images", transforms=ToTensor())
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 6. Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # background + food

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

model.to(device)

# 7. Baseline Inference (Before Training)
model.eval()
images, targets = next(iter(dataloader))
images = [img.to(device) for img in images]

with torch.no_grad():
    outputs_before = model(images)

outputs_before[0]

# 8. Training Loop + Checkpoints
import os
import torch

num_epochs = 10
loss_history = []

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.005, momentum=0.9, weight_decay=0.0005
)

ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "fasterrcnn_food.pt")

start_epoch = 0
if os.path.exists(ckpt_path):
    print(f"Найден чекпоинт: {ckpt_path}, загружаю...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss_history = checkpoint.get("loss_history", [])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Продолжаем с эпохи {start_epoch+1}/{num_epochs}")
else:
    print("Чекпоинт не найден, начинаем обучение с нуля.")

current_epoch = start_epoch

try:
    for epoch in range(start_epoch, num_epochs):
        current_epoch = epoch
        model.train()
        epoch_loss = 0.0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
        }, ckpt_path)
        print(f"Чекпоинт сохранён: {ckpt_path}")

except KeyboardInterrupt:
    print("\nОбучение прервано пользователем. Сохраняю чекпоинт...")
    torch.save({
        "epoch": current_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
    }, ckpt_path)
    print(f"Чекпоинт сохранён: {ckpt_path}")

# 9. Training Progress Visualization
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Progress")
plt.show()

# 10. Inference After Training
model.eval()
with torch.no_grad():
    outputs_after = model(images)

outputs_after[0]
