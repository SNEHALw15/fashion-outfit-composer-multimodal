import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class MultimodalPolyvoreDataset(Dataset):
    def __init__(
        self,
        json_path,
        image_dir,
        item_metadata_path,
        outfit_title_path,
        max_items=8,
        transform=None
    ):
        super().__init__()

        with open(json_path, "r") as f:
            self.data = json.load(f)

        with open(item_metadata_path, "r") as f:
            self.item_metadata = json.load(f)

        # Outfit titles are optional since your JSON doesn't contain set_id
        if os.path.exists(outfit_title_path):
            with open(outfit_title_path, "r") as f:
                self.outfit_titles = json.load(f)
        else:
            self.outfit_titles = {}

        self.image_dir = image_dir
        self.max_items = max_items

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        outfit = self.data[idx]

        item_ids = outfit["items"]
        label = torch.tensor(outfit["label"], dtype=torch.float32)

        images = []
        text_features = []

        for item_id in item_ids[:self.max_items]:

            # -------- IMAGE --------
            image_path = os.path.join(self.image_dir, f"{item_id}.jpg")

            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
            else:
                image = torch.zeros(3, 224, 224)

            images.append(image)

            # -------- TEXT --------
            metadata = self.item_metadata.get(item_id, {})

            title = metadata.get("title", "")
            category = metadata.get("category_name", "")

            text = f"{category} {title}"
            text_features.append(text)

        # Padding if fewer than max_items
        while len(images) < self.max_items:
            images.append(torch.zeros(3, 224, 224))
            text_features.append("")

        images = torch.stack(images)

        # Since you don't have set_id, no outfit title
        outfit_title = ""

        return images, text_features, outfit_title, label