import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, model_config, shared_store):
        super(Seq2SeqTransformer, self).__init__()
        self.shared_store = shared_store
        
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
                tgt: Tensor):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
        
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src: Tensor, tgt: Tensor):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

        src_padding_mask = (src == self.shared_store.special_symbols.index('<pad>')).transpose(0, 1)
        tgt_padding_mask = (tgt == self.shared_store.special_symbols.index('<pad>')).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def greedy_decode(self, src: Tensor, src_mask: Tensor, max_len: int, start_symbol: str) -> Tensor:
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for _ in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.shared_store.special_symbols.index('<eos>'):
                break
        return ys

    def translate(self, src_sentence: str, src_language: str, tgt_language: str) -> str:
        self.eval()
        src = self.shared_store.text_transform[src_language](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5,
                                        start_symbol=self.shared_store.special_symbols.index('<bos>')).flatten()
        return " ".join(self.shared_store.vocab_transform[tgt_language].lookup_tokens(\
            list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
