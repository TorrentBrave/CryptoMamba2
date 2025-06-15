import torch
import torch.nn as nn
from models.layers.Mamba_EncDec import Encoder, EncoderLayer
from models.layers.Embed import DataEmbedding_inverted

from mamba_ssm import Mamba


class SMamba(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, 
                 num_features,
                 seq_len,
                 use_norm,
                 d_model,
                 d_state,
                 d_ff,
                 dropout,
                 activation,
                 e_layers
                 ):
        super(SMamba, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.use_norm = use_norm
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.e_layers = e_layers
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=self.d_model,  # Model dimension d_model
                            d_state=self.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.d_model,  # Model dimension d_model
                            d_state=self.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projector = nn.Linear(self.d_model, 1, bias=True)
        self.projector_features = nn.Linear(self.num_features, 1, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, 1, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, 1, 1))

        output = self.projector_features(dec_out)

        return output
    
    def forward(self, x_enc):
        return self.forecast(x_enc.permute(0, 2, 1))
