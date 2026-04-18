import torch.nn as nn


class CategoryEncoder(nn.Module):
    def __init__(self, num_categories, embedding_dim=128):
        super(CategoryEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim
        )

    def forward(self, category_ids):
        return self.embedding(category_ids)