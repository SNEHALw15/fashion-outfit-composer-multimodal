import torch
import json
import os
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from multimodal_model import MultimodalCompatibilityModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "images"
VAL_JSON = "train_with_negatives.json"
ITEM_METADATA = "polyvore_item_metadata.json"
CHECKPOINT = "checkpoint.pth"

MAX_ITEMS = 8

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔥 FIXED: Load metadata (supports JSONL format)
metadata = {}

with open(ITEM_METADATA, 'r') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            item_id = str(item.get("item_id") or item.get("id"))
            metadata[item_id] = item
        except:
            continue

# Load dataset
with open(VAL_JSON) as f:
    data = json.load(f)

# Load model
model = MultimodalCompatibilityModel().to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

correct = 0
total = 0

# 🔥 For AUC
all_labels = []
all_scores = []

for outfit in data:

    label = outfit["label"]
    item_ids = outfit["items"]

    images = []

    for item_id in item_ids:

        item_id = str(item_id)
        path = os.path.join(IMAGE_DIR, f"{item_id}.jpg")

        if not os.path.exists(path):
            continue

        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)
        except:
            continue

    if len(images) == 0:
        continue

    # Padding
    while len(images) < MAX_ITEMS:
        images.append(torch.zeros(3,224,224))

    images = torch.stack(images).unsqueeze(0).to(DEVICE)

    # 🔥 Text extraction
    texts = []

    for item_id in item_ids:
        item_id = str(item_id)

        if item_id in metadata:
            m = metadata[item_id]

            category = m.get("category", "")
            title = m.get("title", "")

            texts.append(category + " " + title)

    text = " ".join(texts)

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        text_emb = model.text_encoder(**encoded).pooler_output
        output = model(images, text_emb)

    score = output.item()
    pred = 1 if score > 0.5 else 0

    # 🔥 Store for AUC
    all_labels.append(int(label))
    all_scores.append(score)

    if pred == label:
        correct += 1

    total += 1

# 🔥 Compute AUC safely
try:
    auc = roc_auc_score(all_labels, all_scores)
except:
    auc = "Could not compute (only one class present)"

print("\nTotal samples:", total)
print("Correct predictions:", correct)
print("Accuracy:", correct / total if total > 0 else 0)
print("AUC Score:", auc)

# 🔥 Generate ROC Curve
try:
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}" if isinstance(auc, float) else "AUC unavailable")
    plt.plot([0, 1], [0, 1], linestyle='--')  # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("roc_curve.png")
    print("ROC curve saved as roc_curve.png")

except Exception as e:
    print("Could not generate ROC curve:", e)