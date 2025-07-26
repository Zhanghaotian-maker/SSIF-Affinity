import torch
from torch import nn
import torch.nn.functional as F
from model.modules.rope import RotaryPositionEmbedding


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope = RotaryPositionEmbedding(hidden_dim)

    def forward(self, h1, h2):
        batch_size, seq_len1, hidden_dim = h1.size()
        _, seq_len2, _ = h2.size()
        query = self.query_proj(h1.view(batch_size * seq_len1, hidden_dim)).view(batch_size, seq_len1, hidden_dim)
        key = self.key_proj(h2.view(batch_size * seq_len2, hidden_dim)).view(batch_size, seq_len2, hidden_dim)
        value = self.value_proj(h2.view(batch_size * seq_len2, hidden_dim)).view(batch_size, seq_len2, hidden_dim)

        query, key = self.rope(query, key)

        energy = torch.bmm(query, key.permute(0, 2, 1)) / (hidden_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(attention, value)
        out = self.out_proj(out.view(batch_size * seq_len1, hidden_dim)).view(batch_size, seq_len1, hidden_dim)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, transition_dropout=0.0, residual_dropout=0.0, transition_factor=4):
        super().__init__()
        self.cross_attn = CrossAttention(embed_dim)
        self.attn_layer_norm_q = nn.LayerNorm(embed_dim)
        self.attn_layer_norm_kv = nn.LayerNorm(embed_dim)
        self.out_layer_norm = nn.LayerNorm(embed_dim)
        self.transition = nn.Sequential(
            nn.Linear(embed_dim, int(transition_factor * embed_dim)),
            nn.ReLU(),
            nn.Dropout(p=transition_dropout),
            nn.Linear(int(transition_factor * embed_dim), embed_dim)
        )
        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, q, k):
        q_norm = self.attn_layer_norm_q(q)
        k_norm = self.attn_layer_norm_kv(k)
        cross_out = self.cross_attn(q_norm, k_norm)
        
        q = q + self.residual_dropout_1(cross_out)
        
        residual = q
        q = self.out_layer_norm(q)
        q = residual + self.residual_dropout_2(self.transition(q))
        return q
    