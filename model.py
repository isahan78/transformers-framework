# This code defines a Transformer class, which inherits from nn.Module and implements the forward pass of the model. 
# In the constructor, it instantiates the encoder, decoder and normalization and dropout layers. 
# The forward method takes in inputs src, tgt, src_mask, tgt_mask and then it passes the src and src_mask through the encoder, 
# and it passes the tgt, encoder_output and tgt_mask and src_mask through the decoder and finally it returns the output from 
# the decoder after normalizing it.

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from attention import MultiHeadAttention
from ffn import FeedForwardNetwork

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.decoder = Decoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return self.norm(decoder_output)
