import torch
from torch import Tensor


class Processor():
    def __init__(self, model, device, special_symbols):
        self.model = model
        self.device = device
        self.special_symbols = special_symbols
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src: Tensor, tgt: Tensor):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.special_symbols.index('<pad>')).transpose(0, 1)
        tgt_padding_mask = (tgt == self.special_symbols.index('<pad>')).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def greedy_decode(self, src: Tensor, src_mask: Tensor, max_len: int, start_symbol: str, special_symbols) -> Tensor:
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for _ in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == special_symbols.index('<eos>'):
                break
        return ys

    def translate(self, src_sentence: str, tokenizer, special_symbols) -> str:
        self.model.eval()
        src = torch.tensor(tokenizer.encode(src_sentence).ids).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        with torch.no_grad():
            tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 15,
                                            start_symbol=special_symbols.index('<bos>'), 
                                            special_symbols=special_symbols).flatten()
        return tokenizer.decode(list(tgt_tokens.cpu().numpy()))