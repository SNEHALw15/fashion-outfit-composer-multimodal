import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.optim as optim
import os

from multimodal_dataset import MultimodalPolyvoreDataset
from multimodal_model import MultimodalCompatibilityModel


# ----------------------------
# CONFIG
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", DEVICE)

JSON_PATH = "train_with_negatives.json"
IMAGE_DIR = "images"
ITEM_METADATA = "polyvore_item_metadata.json"
OUTFIT_TITLES = "polyvore_outfit_titles.json"

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
MAX_ITEMS = 8

CHECKPOINT_PATH = "checkpoint.pth"

# ----------------------------
# DATASET
# ----------------------------
dataset = MultimodalPolyvoreDataset(
    json_path=JSON_PATH,
    image_dir=IMAGE_DIR,
    item_metadata_path=ITEM_METADATA,
    outfit_title_path=OUTFIT_TITLES,
    max_items=MAX_ITEMS
)

print("✅ Dataset loaded")
print("Dataset size:", len(dataset))

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# MODEL
# ----------------------------
model = MultimodalCompatibilityModel().to(DEVICE)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

start_epoch = 0

# ----------------------------
# LOAD CHECKPOINT IF EXISTS
# ----------------------------
if os.path.exists(CHECKPOINT_PATH):

    print("⚡ Loading checkpoint...")

    checkpoint = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")


# ----------------------------
# TRAIN LOOP
# ----------------------------
print("\n🚀 Starting Training...\n")

for epoch in range(start_epoch, EPOCHS):

    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):

        images, text_features, outfit_title, labels = batch

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        text_features = list(text_features)
        text_features = list(zip(*text_features))

        flattened_text = []

        for outfit in text_features:
            combined = " ".join(outfit)
            flattened_text.append(combined)

        # ---- Tokenize ----
        encoded = tokenizer(
            flattened_text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        # ---- BERT Forward ----
        text_embeddings = model.text_encoder(**encoded).pooler_output

        # ---- Model Forward ----
        outputs = model(images, text_embeddings)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

    # ----------------------------
    # SAVE CHECKPOINT
    # ----------------------------
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, CHECKPOINT_PATH)

    print("💾 Checkpoint saved\n")

print("\n✅ Training Complete")