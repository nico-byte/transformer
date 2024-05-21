import torch
from processor import Processor
from config import TokenizerConfig, SharedConfig
from data import create_text_transform, load_vocab
        

def translate_sequence(run_id, sequence):
    checkpoint = torch.jit.load(f'./results/{run_id}/checkpoint.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_lang = 'de'
    tgt_lang = 'en'
      
    tkn_conf = TokenizerConfig()
    print(tkn_conf.model_dump())
    
    tokenizer = {
            tkn_conf.src_language: tkn_conf.src_tokenizer,
            tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
      }


    shared_config = SharedConfig()
    
    token_transform = tokenizer
    vocab_transform = load_vocab(run_id)
    text_transform = create_text_transform(src_lang, tgt_lang, token_transform, vocab_transform)
    special_symbols = shared_config.special_symbols
        
    translator = Processor(checkpoint, device, special_symbols)
    
    output = translator.translate(sequence, src_language=src_lang, 
          tgt_language=tgt_lang, text_transform=text_transform, vocab_transform=vocab_transform, 
          special_symbols=special_symbols)
    
    print(f'Input: {sequence}, Output: {output}')
    
if __name__ == '__main__':
    sequence = "Eine Gruppe Pinguine steht vor einem Iglu und lacht sich tot ."
    run_id = "multi30k-small"
    translate_sequence(run_id, sequence)