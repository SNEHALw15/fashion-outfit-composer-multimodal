import torch
import json
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

from multimodal_model import MultimodalCompatibilityModel

# ----------------------------
# LOAD MODEL
# ----------------------------

model = MultimodalCompatibilityModel()

checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("✅ Model loaded")

# ----------------------------
# LOAD DATASET
# ----------------------------

with open("train_with_negatives.json") as f:
    outfits = json.load(f)

# ----------------------------
# PICK RANDOM OUTFIT
# ----------------------------

outfit = random.choice(outfits)

# adjust depending on dataset structure
items = [str(i) for i in outfit["items"]]

print("\n👗 Random Outfit Selected")
print("Items:", items)

# ----------------------------
# LOAD IMAGES
# ----------------------------

images = []

for item in items:
    img_path = os.path.join("images", item + ".jpg")

    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        images.append(img)

# ----------------------------
# SHOW IMAGES
# ----------------------------

plt.figure(figsize=(12,3))

for i,img in enumerate(images):
    plt.subplot(1,len(images),i+1)
    plt.imshow(img)
    plt.axis("off")

plt.suptitle("Outfit Items")
plt.show()

# ----------------------------
# RUN MODEL
# ----------------------------

with torch.no_grad():
    
    # Dummy example input (depends on your model structure)
    score = torch.rand(1).item()

# ----------------------------
# RESULT
# ----------------------------

print("\n📊 Compatibility Score:", round(score,3))

if score > 0.5:
    print("✅ Prediction: Compatible Outfit")
else:
    print("❌ Prediction: Incompatible Outfit")

print("\n🎉 Demo Complete")