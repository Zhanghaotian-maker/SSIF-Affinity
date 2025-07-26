import torch
from torch import nn
from model.modules.attention import TransformerBlock


class CoFormer(nn.Module):
    def __init__(self, seq_dim=1280, struct_dim=512, embed_dim=64, num_blocks=6,
                 transition_dropout=0.0, residual_dropout=0.0, transition_factor=4):
        super().__init__()

        self.seq_proj = nn.Linear(seq_dim, embed_dim)
        self.struct_proj = nn.Linear(struct_dim, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, transition_dropout, residual_dropout, transition_factor)
            for _ in range(num_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.norm_dropout = nn.Dropout(0.5)

    def forward(self, h_interface_sequence, h_interface_structure):

        q1 = self.seq_proj(h_interface_sequence.x)
        k1 = self.struct_proj(h_interface_structure.x)
        q1 = q1.unsqueeze(0)
        k1 = k1.unsqueeze(0)
        for block in self.blocks:
            q1 = block(q1, k1)
        q1 = self.final_layer_norm(q1)
        q1 = self.norm_dropout(q1)
        q1 = torch.flatten(q1)

        q2 = self.struct_proj(h_interface_structure.x)
        k2 = self.seq_proj(h_interface_sequence.x)
        q2 = q2.unsqueeze(0)
        k2 = k2.unsqueeze(0)
        for block in self.blocks:
            q2 = block(q2, k2)
        q2 = self.final_layer_norm(q2)
        q2 = self.norm_dropout(q2)
        q2 = torch.flatten(q2)

        h = torch.cat((q1, q2), dim=0)
        return h
    
    