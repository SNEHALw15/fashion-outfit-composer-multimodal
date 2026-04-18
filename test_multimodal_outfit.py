import torch
import json
import os
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from multimodal_model import MultimodalCompatibilityModel


# ----------------------------
# CONFIG
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "images"
ITEM_METADATA = "polyvore_item_metadata.json"
OUTFIT_TITLES = "polyvore_outfit_titles.json"
CHECKPOINT = "checkpoint.pth"

MAX_ITEMS = 8


# ----------------------------
# IMAGE TRANSFORM
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# ----------------------------
# LOAD METADATA
# ----------------------------

with open(ITEM_METADATA) as f:
    item_metadata = json.load(f)

with open(OUTFIT_TITLES) as f:
    outfit_titles = json.load(f)


# ----------------------------
# LOAD MODEL
# ----------------------------

model = MultimodalCompatibilityModel().to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("✅ Model loaded")


# ----------------------------
# INPUT OUTFIT
# ----------------------------

# Ask user for item IDs
user_input = input("Enter item IDs separated by comma: ")

items = [item.strip() for item in user_input.split(",")]

print("\nOutfit items:")
for item in items:
    print(item)

# ----------------------------
# LOAD IMAGES
# ----------------------------

image_tensors = []

for item_id in items:

    image_path = os.path.join(IMAGE_DIR, item_id + ".jpg")

    img = Image.open(image_path).convert("RGB")
    img = transform(img)

    image_tensors.append(img)

# padding
while len(image_tensors) < MAX_ITEMS:
    image_tensors.append(torch.zeros(3,224,224))

images = torch.stack(image_tensors).unsqueeze(0).to(DEVICE)


# ----------------------------
# LOAD TEXT
# ----------------------------

text_list = []

for item_id in items:

    meta = item_metadata[item_id]

    category = meta.get("category", "")
    title = meta.get("title", "")

    text = category + " " + title
    text_list.append(text)

combined_text = " ".join(text_list)


encoded = tokenizer(
    [combined_text],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

encoded = {k:v.to(DEVICE) for k,v in encoded.items()}

text_embeddings = model.text_encoder(**encoded).pooler_output


# ----------------------------
# PREDICTION
# ----------------------------

with torch.no_grad():

    output = model(images, text_embeddings)

score = output.item()

print("\nOutfit items:")
for i in items:
    print(i)

print("\nCompatibility Score:", score)