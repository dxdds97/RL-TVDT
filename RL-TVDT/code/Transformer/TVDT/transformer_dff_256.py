# from this import d
import torch
import torch.nn as nn

# sys.path.append('Transformer/')
from TVDT.two_stage_attn import EncoderLayer, DecoderLayer, Encoder, Decoder
from TVDT.attn import FullAttention, AttentionLayer
from TVDT.embed import DataEmbedding
from TVDT.variable_embed import DSW_embedding
from einops import rearrange


class Transformer_base(nn.Module):
    def __init__(self, enc_in, dec_in, c_out,
                 d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256,
                 dropout=0.0, activation='gelu', output_attention=False):
        super(Transformer_base, self).__init__()
        seg_len = 6  # 5*12 6*10 4*15
        seq_len = 60
        data_dim = enc_in
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # cross embed
        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (seq_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, seg_len)
        self.linear_reduce = nn.Linear(d_model, seg_len)
        self.linear_dec = nn.Linear(enc_in, d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection_decoder = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # enc_out = self.enc_embedding(x_enc)
        x_seq = self.enc_value_embedding(x_enc)  # linear   ->   b d seg_num d_model  seg_len=6
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        dec_out = self.dec_embedding(x_dec)

        enc_out = self.encoder(x_seq, attn_mask=enc_self_mask)
        eo = rearrange(enc_out, 'b ts_d seg_num d_model -> (b ts_d seg_num) d_model', seg_num=10)
        lr = self.linear_reduce(eo)
        re = rearrange(lr, '(b ts_d seg_num) seg_len -> b (seg_len seg_num) ts_d',seg_num=10,ts_d=10)
        enc_out = self.linear_dec(re)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        output = self.projection_decoder(dec_out)

        return enc_out, dec_out, output


