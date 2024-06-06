import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
 
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self, model_config):
        super(Seq2SeqTransformer, self).__init__()
        
        self.transformer = Transformer(d_model=model_config.emb_size,
                                       nhead=model_config.nhead,
                                       num_encoder_layers=model_config.num_encoder_layers,
                                       num_decoder_layers=model_config.num_decoder_layers,
                                       dim_feedforward=model_config.dim_feedforward,
                                       dropout=model_config.dropout)
        self.generator = nn.Linear(model_config.emb_size, model_config.tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(model_config.src_vocab_size, model_config.emb_size)
        self.tgt_tok_emb = TokenEmbedding(model_config.tgt_vocab_size, model_config.emb_size)
        self.positional_encoding = PositionalEncoding(
            model_config.emb_size, dropout=model_config.dropout)
                                
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: Tensor,
                tgt: Tensor, 
                src_mask, 
                tgt_mask, 
                src_padding_mask, 
                tgt_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
        return self.generator(outs)

    @torch.jit.export
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    @torch.jit.export
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
