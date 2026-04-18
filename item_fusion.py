import torch
import torch.nn as nn


class ItemFusion(nn.Module):
    def __init__(self,
                 image_dim=1408,
                 category_dim=128,
                 text_dim=768,
                 output_dim=512):

        super(ItemFusion, self).__init__()

        input_dim = image_dim + category_dim + text_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )

    def forward(self, image_feat, category_feat, text_feat):
        # Concatenate
        x = torch.cat([image_feat, category_feat, text_feat], dim=-1)
        return self.mlp(x)