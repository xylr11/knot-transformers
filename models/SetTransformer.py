import torch
from torch import nn
from torch.nn import functional as F

class MAB(nn.Module):
    """Multihead Attention Block with pre-norm + dropout."""
    def __init__(self, dim_Q, dim_KV, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.fc_q = nn.Linear(dim_Q, hidden_dim)
        self.fc_k = nn.Linear(dim_KV, hidden_dim)
        self.fc_v = nn.Linear(dim_KV, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.ln0 = nn.LayerNorm(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # expand
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, Q, K, key_mask=None):
        # project
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)

        # pre-norm attention
        attn_out, _ = self.mha(
            self.ln0(Q_proj),
            self.ln0(K_proj),
            self.ln0(V_proj),
            key_padding_mask=key_mask
        )
        H = Q_proj + attn_out

        # pre-norm feedforward
        H = H + self.ff(self.ln1(H))
        return H

class SAB(nn.Module):
    """Self-Attention Block: SAB(X) = MAB(X,X)."""
    def __init__(self, dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(dim, dim, hidden_dim, num_heads, dropout=dropout)

    def forward(self, X, key_mask=None):
        return self.mab(X, X, key_mask=key_mask)

class Encoder(nn.Module):
    """Encodes input points with mask (pre-norm, variance-preserving)."""
    def __init__(self, dim_input=2, hidden_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

        self.input_proj = nn.Sequential(
            nn.Linear(dim_input, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList([
            SAB(hidden_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, mask=None, log_diversity=False):
        key_mask = (~mask.bool()) if mask is not None else None
        self.scale = nn.Parameter(torch.ones(1) * 3.0)
        H = self.input_proj(X * self.scale)

        layer_stats = []
        for i, layer in enumerate(self.layers):
            H = layer(H, key_mask)
            if log_diversity:
                var_set = H.var(dim=1).mean()   # variance across set elems
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
    """
    Permutation-invariant decoder that uses PMA but reduces collapse by:
      - concatenating a global set summary (mean) to each query
      - removing final LayerNorm in the post-MLP
      - a small learnable scale on queries to restore magnitude if needed

    Inputs:
      H : (B, N, D)  -- encoder outputs (keys/values)
      mask : (B, N)  -- boolean mask with 1=valid, 0=pad (or None)
    Returns:
      out : (B, m, 3)  -- (x, y, p) per output
      stats : list or None  -- diagnostic stats if log_diversity True
    """
    def __init__(self, hidden_dim=128, num_heads=4, num_outputs=64, refine_layers=1):
        super().__init__()
        self.num_outputs = num_outputs
        self.m = max([num_outputs, hidden_dim // 2])
        self.pma = PMA(hidden_dim, hidden_dim, num_heads, m=self.m)
        self.refine = nn.ModuleList([SAB(hidden_dim, hidden_dim, num_heads)
                                     for _ in range(refine_layers)])
        # post: concat(query, global_mean) => 2D -> hidden -> 3
        self.post = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        # small trainable scale applied to queries (helps undo LN shrinkage)
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, H, mask=None, log_diversity=False):
        """
        H: (B, N, D)
        mask: (B, N) with 1=valid, 0=pad or None
        """
        # key_mask passed to MHA expects True = ignore/pad
        key_mask = (~mask.bool()) if mask is not None else None

        # PMA: S (queries) attend to H (keys/values)
        Q = self.pma(H, key_mask=key_mask)   # (B, m, D)

        # optional refinement among queries (SAB: equivariant w.r.t. query order)
        for sab in self.refine:
            Q = sab(Q)   # (B, m, D)

        # global, permutation-invariant summary of H
        g = H.mean(dim=1, keepdim=True)               # (B, 1, D)
        g = g.expand(-1, self.m, -1)         # (B, m, D)

        # concat scaled Q with global summary -> (B, m, 2D)
        Qcat = torch.cat([Q * self.scale, g], dim=-1)

        out = self.post(Qcat)                    # (B, m, 3)

        out = out[:, :self.num_outputs] # (B, num_outputs, 3)

        out = torch.cat([out[..., :2], torch.sigmoid(out[..., 2:3])], dim=-1)

        if log_diversity:
            # diagnostics based on the *raw* queries Q (before concat/post)
            var_within = Q.var(dim=1).mean()    # variance across queries (m)
            var_batch = Q.var(dim=0).mean()     # variance across batch for each feature
            stats = [{
                "decoder_layer": "pma+refine+global-resid",
                "var_queries": var_within.detach().cpu(),
                "var_batch": var_batch.detach().cpu(),
                "var_out": out.var(dim=1).mean().detach().cpu()
            }]
            return out, stats

        return out, None

class SetTransformer(nn.Module):
    """
    Full SetTransformer using your Encoder and PMADecoderLessCollapse.

    forward(X, mask=None, log_diversity=False) -> (Y, stats)
      - X: (B, N, dim_input)
      - mask: (B, N) boolean (1=valid, 0=pad) or None
      - Y: (B, num_outputs, 3)
      - stats: dict with keys 'encoder' and 'decoder' when log_diversity True
    """
    def __init__(self,
                 dim_input=2,
                 hidden_dim=256,
                 num_outputs=64,
                 num_heads=4,
                 num_encoder_layers=3,
                 decoder_refine_layers=3):
        super().__init__()
        self.encoder = Encoder(dim_input=dim_input,
                               hidden_dim=hidden_dim,
                               num_heads=num_heads,
                               num_layers=num_encoder_layers
                               )

        self.decoder = PMADecoder(hidden_dim=hidden_dim,
                                              num_heads=num_heads,
                                              num_outputs=num_outputs,
                                              refine_layers=decoder_refine_layers)

    def forward(self, X, mask=None, log_diversity=False):
        """
        X: (B, N, dim_input)
        mask: (B, N) with 1=valid, 0=pad  (or None)
        """
        # Ask encoder for stats iff log_diversity requested
        H, enc_stats = self.encoder(X, mask, log_diversity=log_diversity)

        # Decoder needs the mask for pooling (keys/values are H of length N)
        Y, dec_stats = self.decoder(H, mask=mask, log_diversity=log_diversity)

        if log_diversity:
            stats = {"encoder": enc_stats, "decoder": dec_stats}
            return Y, stats

        return Y, None