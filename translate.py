import torch
from torch import Tensor
from config import TokenizerConfig, SharedStore
from data import create_text_transform, load_vocab
from config import TransformerConfig
from transformer import Seq2SeqTransformer


class Translate():
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

    def translate(self, src_sentence: str, src_language: str, tgt_language: str, 
                  text_transform, vocab_transform, special_symbols) -> str:
        self.model.eval()
        src = text_transform[src_language](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        with torch.no_grad():
            tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5,
                                            start_symbol=special_symbols.index('<bos>'), 
                                            special_symbols=special_symbols).flatten()
        return " ".join(vocab_transform[tgt_language].lookup_tokens(\
            list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        

def translate_sequence(run_id, sequence):
    checkpoint = torch.jit.load(f'./results/{run_id}/checkpoint.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_lang = 'de'
    tgt_lang = 'en'
      
    tkn_conf = TokenizerConfig()
    print(tkn_conf.model_dump())


    shared_store = SharedStore(
        token_transform={
              tkn_conf.src_language: tkn_conf.src_tokenizer,
              tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
        }
    )
    
    token_transform = shared_store.token_transform
    vocab_transform = load_vocab(run_id)
    text_transform = create_text_transform(src_lang, tgt_lang, token_transform, vocab_transform)
    special_symbols = shared_store.special_symbols
        
    translator = Translate(checkpoint, device, special_symbols)
    
    print(f'Input: {sequence}, Output: {translator.translate(sequence, src_language=src_lang, 
          tgt_language=tgt_lang, text_transform=text_transform, vocab_transform=vocab_transform, 
          special_symbols=special_symbols)}')
    
if __name__ == '__main__':
    sequence = "Ein Mann mit blonden Haar hat ein Haus aus Steinen gebaut ."
    run_id = 1234
    translate_sequence(run_id, sequence)