import torch
from torch import nn

class SumMaxPooling(nn.Module):
    """
    Concat(Sum, Max) pooling for sets with masks.
    
    Input:
        x: (batch, set_size, d)
        mask: (batch, set_size), 1 = valid, 0 = pad
    Output:
        pooled: (batch, 2d)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        mask = mask.bool()

        # sum pooling
        sum_pool = torch.sum(x * mask.unsqueeze(-1), dim=1)

        # max pooling (mask padded elements with -inf)
        masked_x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_pool, _ = masked_x.max(dim=1)

        # handle case: all elements masked (avoid -inf)
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        return torch.cat([sum_pool, max_pool], dim=-1)

class DeepSets(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, num_outputs=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pooling = SumMaxPooling()
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs * 3)  # (x, y, confidence)
        )
        self.num_outputs = num_outputs

        with torch.no_grad():
            # spread outputs roughly on unit circle
            angle = torch.linspace(0, 2 * torch.pi, steps=num_outputs + 1)[:-1]
            init_xy = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)  # (num_outputs, 2)
            init_xy = init_xy.flatten()  # shape (2*num_outputs,)

            final_linear = self.decoder[-1]
            # set x,y biases
            final_linear.bias[:2 * num_outputs].copy_(init_xy)
            # set conf bias (start low, so confidence ramps up only when useful)
            final_linear.bias[2 * num_outputs:].fill_(-2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, x, mask, log_diversity=False):
        encoded = self.encoder(x)                        # (B, L, hidden_dim)
        pooled = self.pooling(encoded, mask)             # (B, 2 * hidden_dim)
        decoded = self.decoder(pooled)                   # (B, num_outputs * 3)
        decoded = decoded.view(-1, self.num_outputs, 3)  # (B, M, 3)
        decoded[..., 2] = decoded[..., 2].sigmoid()
        return decoded, None # The None here is to match the returns of SetTransformers which also returns stats