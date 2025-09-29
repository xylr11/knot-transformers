import torch
from torch import nn

class DeepSets(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, num_outputs=49):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pooling = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1)        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs * 3)  # (x, y, confidence)
        )
        self.num_outputs = num_outputs

    def forward(self, x, mask):
        encoded = self.encoder(x)                        # (B, L, hidden_dim)
        pooled = self.pooling(encoded, mask)             # (B, hidden_dim)
        decoded = self.decoder(pooled)                   # (B, num_outputs * 3)
        decoded = decoded.view(-1, self.num_outputs, 3)  # (B, M, 3)
        decoded[..., 2] = decoded[..., 2].sigmoid() 
        return decoded, None