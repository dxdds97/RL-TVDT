import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt
from cross_models.attn import AttentionLayer, FullAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout, activation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x

class MultiheadFeedForward(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, activation):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mhfw = nn.ModuleList([FeedForward(d_model=self.head_dim, ff_dim=ff_dim, dropout=dropout, activation=activation) for i in range(self.n_heads)])

    def forward(self, x): # [bs, seq_len, d_model]
        bs = x.shape[0]
        input = x.reshape(bs, -1, self.n_heads, self.head_dim) # [bs, seq_len, n_heads, head_dim]
        outputs = []
        for i in range(self.n_heads):
            outputs.append(self.mhfw[i](input[:, :, i, :])) # [bs, seq_len, head_dim]
        outputs = torch.stack(outputs, dim=-2).reshape(bs, -1, self.d_model) # [bs, seq_len, n_heads, head_dim]
        return outputs


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, n_heads=8, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 256
        seg_num = 10
        factor = 10
        self.attention = attention
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)  # 时间维度的注意力层

        self.mhfw = MultiheadFeedForward(d_model=d_model, n_heads=n_heads, ff_dim=d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)  # 发送维度的注意力层
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)  # 接收维度的注意力层
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))  # 路由器参数
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None):
        # time stage
        batch = int(x.shape[0])  # 1408 10 10 128
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')  # 14080 10 128
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages
        # to build the D-to-D connection
        # 维度阶段：使用一组可学习的向量来聚合和分发信息以构建 D-to-D 的连接
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)  # 14080 10 128
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model',
                              repeat=batch)  # 10 10 128
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        # 14080 10 128 -> 1408 10 10 128
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out





class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x= attn_layer(x, attn_mask=attn_mask)
                # attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff, n_heads=8,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.mhfw = MultiheadFeedForward(d_model=d_model, n_heads=n_heads, ff_dim=d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.mhfw(y)

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x