import torch
from torch import nn

class MAB(nn.Module):
    """Multihead Attention Block with mask support."""
    def __init__(self, dim_Q, dim_KV, hidden_dim, num_heads):
        super().__init__()
        self.fc_q = nn.Linear(dim_Q, hidden_dim)
        self.fc_k = nn.Linear(dim_KV, hidden_dim)
        self.fc_v = nn.Linear(dim_KV, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln0 = nn.LayerNorm(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, Q, K, key_mask=None):
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)
        attn_output, _ = self.mha(Q_proj, K_proj, V_proj, key_padding_mask=key_mask)
        H = self.ln0(Q_proj + attn_output)
        H = self.ln1(H + self.ff(H))
        return H

class ISAB(nn.Module):
    """Induced Set Attention Block with mask support."""
    def __init__(self, dim_in, hidden_dim, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inds, hidden_dim))
        self.mab1 = MAB(hidden_dim, dim_in, hidden_dim, num_heads)
        self.mab2 = MAB(dim_in, hidden_dim, hidden_dim, num_heads)

    def forward(self, X, mask=None):
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X, key_mask=mask)
        return self.mab2(X, H)

class Encoder(nn.Module):
    """Encodes input points with mask."""
    def __init__(self, dim_in=2, hidden_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            ISAB(dim_in if i == 0 else hidden_dim, hidden_dim, num_heads, num_inds=16)
            for i in range(num_layers)
        ])

    def forward(self, X, mask):
        H = X
        key_mask = ~mask.bool()
        for layer in self.layers:
            H = layer(H, key_mask)
        return H

class CrossDecoder(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_outputs=64, num_layers=3):
        super().__init__()
        self.num_outputs = num_outputs
        self.slot_emb = nn.Parameter(torch.randn(num_outputs, hidden_dim))
        self.query_gen = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MAB(hidden_dim, hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, H, mask):
        pooled = H.mean(dim=1)
        base_query = self.query_gen(pooled)
        Q = base_query.unsqueeze(1) + self.slot_emb.unsqueeze(0)

        key_mask = ~mask.bool()
        for layer in self.layers:
            Q = layer(Q, H, key_mask)

        out = self.post(Q)
        out_xy = out[..., :2]
        out_conf = torch.sigmoid(out[..., 2:3])
        return torch.cat([out_xy, out_conf], dim=-1)

class SetTransformer(nn.Module):
    """Full set-to-set predictor."""
    def __init__(self, dim_input=2, hidden_dim=512, num_outputs=64, num_heads=4, num_layers=6):
        super().__init__()
        self.encoder = Encoder(dim_input, hidden_dim, num_heads, num_layers)
        self.decoder = CrossDecoder(hidden_dim, num_heads, num_outputs, num_layers)

    def forward(self, X, mask):
        H = self.encoder(X, mask)
        Y = self.decoder(H, mask)
        return Y