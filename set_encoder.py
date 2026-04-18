import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    def __init__(self,
                 input_dim=512,
                 num_heads=4,
                 ff_hidden_dim=1024):

        super(SetEncoder, self).__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # LayerNorm after attention
        self.norm1 = nn.LayerNorm(input_dim)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, input_dim)
        )

        # LayerNorm after feedforward
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, item_embeddings, item_mask):
        """
        item_embeddings: (batch_size, max_items, 512)
        item_mask:       (batch_size, max_items)
                         1 = real item
                         0 = padded item
        """

        # key_padding_mask expects True where we want to IGNORE
        key_padding_mask = (item_mask == 0)

        # Self-attention
        attn_output, _ = self.attention(
            item_embeddings,
            item_embeddings,
            item_embeddings,
            key_padding_mask=key_padding_mask
        )

        # Residual + LayerNorm
        x = self.norm1(item_embeddings + attn_output)

        # Feedforward
        ff_output = self.feedforward(x)

        # Residual + LayerNorm
        x = self.norm2(x + ff_output)

        # ----- Masked Mean Pooling -----
        mask = item_mask.unsqueeze(-1)  # (batch, max_items, 1)
        x = x * mask                   # zero-out padded items

        summed = x.sum(dim=1)          # (batch, 512)
        counts = mask.sum(dim=1)       # (batch, 1)

        # Avoid division by zero
        outfit_embedding = summed / counts.clamp(min=1)

        return outfit_embedding  # (batch_size, 512)