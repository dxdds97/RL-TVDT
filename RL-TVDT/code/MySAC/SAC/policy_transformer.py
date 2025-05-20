from torch import nn
import torch
import itertools

from TVDT.attn import AttentionLayer, FullAttention


class policy_transformer_stock_atten2(nn.Module): # attention(long, short), attention(hybrid, relational) 
    def __init__(self, d_model=128, n_heads=4, dropout=0.0, lr=0.0001, output_attention=False, device='cuda:0'):
        super().__init__()
        self.attention = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.attention2 = AttentionLayer(FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.optimizer = torch.optim.Adam(itertools.chain(self.attention.parameters(), self.attention2.parameters()), lr=lr)
        self.device = device

        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习的加权参数
        self.beta = nn.Parameter(torch.tensor(1.0))  # 可学习的加权参数
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 可学习的加权参数

    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, holding, mask=None):
        # relational_feature shape [B, N, D]
        # temporal_feature_short=temporal_feature_long shape [B, N, D]
        # holding shape [B, N] or None
        # return feature shape [B, N, D+1]

        temporal_hybrid_feature, attn = self.attention(
            temporal_feature_long, temporal_feature_short, temporal_feature_short,
            attn_mask=mask
        )
        # temporal_feature_long = self.alpha * temporal_feature_long + self.dropout(temporal_hybrid_feature)  # a
        temporal_feature_long = self.alpha * temporal_feature_long + (1-self.alpha)*self.dropout(temporal_hybrid_feature)  # a
        # temporal_feature_long = temporal_feature_long + self.dropout(temporal_hybrid_feature)  # a
        temporal_feature = self.norm(temporal_feature_long)

        temporal_relational_hybrid_feature, attn = self.attention2(
            temporal_feature, relational_feature, relational_feature,
            attn_mask=mask
        )

        # temporal_feature = self.beta * temporal_feature + self.dropout(temporal_relational_hybrid_feature)  # b
        temporal_feature = self.beta * temporal_feature + (1-self.beta)*self.dropout(temporal_relational_hybrid_feature)  # b
        # temporal_feature = temporal_feature + self.dropout(temporal_relational_hybrid_feature)  # b
        hybrid_feature = self.norm(temporal_feature)

        combined_feature = torch.cat((hybrid_feature, holding), dim=-1)  # [B, N, D+1]

        return combined_feature






