import torch.nn as nn
from torchvision.models import efficientnet_b2


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()

        self.model = efficientnet_b2(pretrained=pretrained)

        # Remove classification head
        self.model.classifier = nn.Identity()

    def forward(self, images):
        return self.model(images)  # (batch*num_items, 1408)