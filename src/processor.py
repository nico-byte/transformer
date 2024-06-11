import torch
from torch import Tensor
import tokenizers


class Processor():
    @classmethod
    def from_instance(cls, model, tokenizer, device):
        processor = cls()
        
        processor.model = model
        processor.tokenizer = tokenizer
        processor.device = device
        
        return processor
        
    @classmethod
    def from_checkpoint(cls, model_checkpoint, tokenizer, device):
        processor = cls()

        processor.model = torch.jit.load(model_checkpoint, map_location=device)
        processor.tokenizer = tokenizers.Tokenizer.from_file(tokenizer)
        processor.device = device
        
        return processor
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square subsequent mask of size sz x sz.

        Args:
            sz (int): Size of the square mask.

        Returns:
            torch.Tensor: Square subsequent mask.
        """

        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src: Tensor, tgt: Tensor, pad_id: int=58100):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == pad_id).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_id).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def greedy_decode(self, src: Tensor, src_mask: Tensor, max_len: int, eos_token_id: int) -> Tensor:
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).type(torch.long).to(self.device)
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
            if next_word == eos_token_id:
                break
        return ys

    def translate(self, src_sentence: str) -> str:
        self.model.eval()
        encoded_src = self.tokenizer.encode(src_sentence).ids
        src = torch.tensor(encoded_src).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        with torch.no_grad():
            tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5,
                                            eos_token_id=2).flatten()
        return self.tokenizer.decode(list(tgt_tokens.cpu().numpy()), skip_special_tokens=True)