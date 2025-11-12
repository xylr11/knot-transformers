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
    """Self-Attention Block: SAB(X) = MAB(X,X)"""
    def __init__(self, dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(dim, dim, hidden_dim, num_heads, dropout=dropout)

    def forward(self, X, key_mask=None):
        return self.mab(X, X, key_mask=key_mask)

class Encoder(nn.Module):
    """Encodes input points with mask"""
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
                var_set = H.var(dim=1).mean()   # variance across zeros
                var_batch = H.var(dim=0).mean() # variance across batch
                layer_stats.append({
                    "layer": i,
                    "var_set": var_set.detach().cpu(),
                    "var_batch": var_batch.detach().cpu()
                })

        return (H, layer_stats) if log_diversity else (H, None)
    
class PMA(nn.Module):
    """Pooling by Multihead Attention with m seeds (see the excellent paper of J. Lee et al.)"""
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
    Inputs:
      H (B, N, D) : encoder outputs (keys/values)
      mask (B, N) : boolean mask with 1=valid, 0=padding from collate (or None)
    Returns:
      out : (B, m, 3) (re, im, conf) per output
      stats : list or None Diagnostic stats if log_diversity True (Warning: probably broken due to changes in training :()
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
        
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, H, mask=None, log_diversity=False):
        """
        H: (B, N, D)
        mask: (B, N) with 1=valid, 0=pad or None
        """
        # key_mask passed to MHA expects True = ignore/pad
        key_mask = (~mask.bool()) if mask is not None else None
        Q = self.pma(H, key_mask=key_mask)
        for sab in self.refine:
            Q = sab(Q)

        g = H.mean(dim=1, keepdim=True)
        g = g.expand(-1, self.m, -1)
        Qcat = torch.cat([Q * self.scale, g], dim=-1)
        out = self.post(Qcat)
        out = out[:, :self.num_outputs]
        out = torch.cat([out[..., :2], torch.sigmoid(out[..., 2:3])], dim=-1)

        if log_diversity:
            # diagnostics
            var_within = Q.var(dim=1).mean()
            var_batch = Q.var(dim=0).mean()
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
    Full Set Transformer

    forward(X, mask=None, log_diversity=False) -> (Y, stats)
      - X: (B, N, dim_input)
      - mask: (B, N) boolean (1=valid, 0=pad) or None
      - Y: (B, num_outputs, 3)
      - stats: dict with keys 'encoder' and 'decoder' when log_diversity True
    
    TODO: this needs to be updated to use the biases from the deepset version, since bad initialization seems to be problematic
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
        # Ask encoder for stats iff log_diversity requested (else None is returned)
        H, enc_stats = self.encoder(X, mask, log_diversity=log_diversity)

        # Decoder needs the mask for pooling (keys/values are H of length N)
        Y, dec_stats = self.decoder(H, mask=mask, log_diversity=log_diversity)

        if log_diversity:
            stats = {"encoder": enc_stats, "decoder": dec_stats}
            return Y, stats

        return Y, None