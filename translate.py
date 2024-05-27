import torch
from processor import Processor
from config import TokenizerConfig, SharedConfig
from data import create_text_transform, load_vocab
from t5_inference import get_base_model, t5_inference


def translate_sequence_from_checkpoint(checkpoint, vocab, sequence, device):    
    checkpoint = torch.jit.load(checkpoint, map_location=device)
    src_lang = 'en'
    tgt_lang = 'de'
      
    tkn_conf = TokenizerConfig()
    
    tokenizer = {
            tkn_conf.src_language: tkn_conf.src_tokenizer,
            tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
      }


    shared_config = SharedConfig()
    
    token_transform = tokenizer
    vocab_transform = load_vocab(vocab)
    text_transform = create_text_transform(src_lang, tgt_lang, token_transform, vocab_transform)
    special_symbols = shared_config.special_symbols
        
    translator = Processor(checkpoint, device, special_symbols)
    
    output = translator.translate(sequence, src_language=src_lang, 
          tgt_language=tgt_lang, text_transform=text_transform, vocab_transform=vocab_transform, 
          special_symbols=special_symbols)
    
    return output    
    

def check_device(dvc=None):
    if dvc is not None:
        try:
            device = torch.device(dvc)
            return device
        except RuntimeError as e:
            print(e)
            print(f'Device {dvc} is not available. Defaulting to CPU.')
            return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
