import torch
from torch import nn
from torch.nn import functional as F

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
    
class SAB(nn.Module):
    """Self-Attention Block: SAB(X) = MAB(X, X)."""
    def __init__(self, dim, hidden_dim, num_heads):
        super().__init__()
        self.mab = MAB(dim, dim, hidden_dim, num_heads)
    def forward(self, X, key_mask=None):
        return self.mab(X, X, key_mask=key_mask)

class ISAB(nn.Module):
    """Induced Set Attention Block with mask support."""
    def __init__(self, dim_in, hidden_dim, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inds, hidden_dim))
        self.mab1 = MAB(hidden_dim, dim_in, hidden_dim, num_heads)
        self.mab2 = MAB(dim_in, hidden_dim, hidden_dim, num_heads)

    def forward(self, X, key_mask=None):
        if key_mask is not None and key_mask.all(dim=1).any():
            raise ValueError("ISAB.forward: all keys masked for some batch elements")
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X, key_mask=key_mask)
        return self.mab2(X, H, key_mask=None)

class Encoder(nn.Module):
    """Encodes input points with mask."""
    def __init__(self, dim_in=2, hidden_dim=128, num_heads=4, num_layers=3, num_inds=32):
        super().__init__()
        self.layers = nn.ModuleList([
            ISAB(dim_in if i == 0 else hidden_dim, hidden_dim, num_heads, num_inds=num_inds)
            for i in range(num_layers)
        ])

    def forward(self, X, mask, log_diversity=False):
        if mask is None:
            key_mask = None
        else:
            key_mask = ~mask.bool()

        H = X

        layer_stats = []
        for i, layer in enumerate(self.layers):
            H = layer(H, key_mask)
            if log_diversity:
                var_set = H.var(dim=1).mean()   # variance across set elements
                var_batch = H.var(dim=0).mean() # variance across batch
                layer_stats.append({
                    "layer": i,
                    "var_set": var_set.detach().cpu(),
                    "var_batch": var_batch.detach().cpu()
                })

        return (H, layer_stats) if log_diversity else (H, None)

class PMA(nn.Module):
    """Pooling by Multihead Attention with m seeds."""
    def __init__(self, dim, hidden_dim, num_heads, m):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, m, dim))
        self.mab = MAB(dim, dim, hidden_dim, num_heads)
    def forward(self, X, key_mask=None):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)        # (B, m, dim)
        return self.mab(S, X, key_mask=key_mask)

class PMADecoder(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_outputs=64, refine_layers=1):
        super().__init__()
        self.num_outputs = num_outputs
        self.pma = PMA(hidden_dim, hidden_dim, num_heads, m=num_outputs)
        self.refine = nn.ModuleList([SAB(hidden_dim, hidden_dim, num_heads)
                                     for _ in range(refine_layers)])
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, H, mask=None, log_diversity=False):
        if mask is not None:
            key_mask = ~mask.bool()
        else:
            key_mask = None

        Q = self.pma(H, key_mask)                 # (B, num_outputs, D)
        for sab in self.refine:
            Q = sab(Q)                            # optional refinement among outputs

        out = self.post(Q)                        # (B, num_outputs, 3)
        out = torch.cat([out[..., :2], torch.sigmoid(out[..., 2:3])], dim=-1)

        if log_diversity:
            var_within = Q.var(dim=1).mean()
            var_batch = Q.var(dim=0).mean()
            stats = [{"decoder_layer": "pma+refine",
                      "var_queries": var_within.detach().cpu(),
                      "var_batch": var_batch.detach().cpu()}]
            return (out, stats)
        return (out, None)
    
class SetTransformer(nn.Module):
    """Set-to-set predictor with less pooling collapse."""
    def __init__(self, dim_input=2, hidden_dim=256, num_outputs=64, num_heads=4, num_layers=4):
        super().__init__()
        self.encoder = Encoder(dim_input, hidden_dim, num_heads, num_layers)

        # Instead of pooling everything into m seeds,
        # project all set elements, then use SAB to refine.
        self.refine = nn.ModuleList([
            SAB(hidden_dim, hidden_dim, num_heads) for _ in range(2)
        ])

        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Keep permutation invariance by selecting num_outputs elements, in particular, no more PMA!
        # I want to try something like pool into L-norms, then use SAB to refine
        self.num_outputs = num_outputs

    def forward(self, X, mask=None, log_diversity=False):
        H, layer_stats = self.encoder(X, mask, log_diversity=log_diversity)

        # Refine the full set embeddings
        for sab in self.refine:
            H = sab(H, key_mask=(~mask.bool() if mask is not None else None))

        # Simple random projection: pick `num_outputs` representatives
        # (Permutation-invariant because it's uniform over set)
        if H.size(1) >= self.num_outputs:
            # deterministic subsample
            Q = H[:, :self.num_outputs, :]
        else:
            # pad if fewer than requested
            pad = self.num_outputs - H.size(1)
            Q = torch.cat([H, H[:, :pad, :]], dim=1)

        out = self.post(Q)
        out = torch.cat([out[..., :2], torch.sigmoid(out[..., 2:3])], dim=-1)

        if log_diversity:
            var_within = Q.var(dim=1).mean()
            var_batch = Q.var(dim=0).mean()
            stats = {"encoder": layer_stats,
                     "decoder": [{"decoder_layer": "variance-preserving",
                                  "var_queries": var_within.detach().cpu(),
                                  "var_batch": var_batch.detach().cpu()}]}
            return (out, stats)
        return (out, None)