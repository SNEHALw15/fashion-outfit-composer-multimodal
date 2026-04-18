import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel


class MultimodalCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- IMAGE BACKBONE --------
        self.image_encoder = models.efficientnet_b2(pretrained=True)
        self.image_encoder.classifier = nn.Identity()
        image_feature_dim = 1408
        # ⚡ Freeze EfficientNet to speed up training
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        print("⚡ EfficientNet frozen. Only BERT + fusion layers will train.")

        # -------- TEXT ENCODER (BERT) --------
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_feature_dim = 768

        # -------- FUSION LAYER --------
        self.fusion = nn.Linear(image_feature_dim + text_feature_dim, 512)

        # -------- COMPATIBILITY HEAD --------
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, images, text_embeddings):

        batch_size, max_items, C, H, W = images.shape
        images = images.view(batch_size * max_items, C, H, W)

        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, max_items, -1)
        image_features = image_features.mean(dim=1)

        fused = torch.cat([image_features, text_embeddings], dim=1)
        fused = self.fusion(fused)

        output = self.classifier(fused)

        return output.squeeze(1)