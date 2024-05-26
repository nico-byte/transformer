import torch
from processor import Processor
from config import TokenizerConfig, SharedConfig
from data import create_text_transform, load_vocab
from t5_inference import get_base_model, t5_inference


def translate_sequence_from_checkpoint(run_id, sequence, device):    
    checkpoint = torch.jit.load(f'./results/{run_id}/checkpoint.pt')
    checkpoint.to(device)
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
    
    
def translate_sequence_from_t5(sequence, device):
    model, transform, sequence_generator = get_base_model()
    output = t5_inference(model, transform, sequence_generator, sequence, device)
    print(output)
    

def check_device(dvc=None):
    if dvc is not None:
        try:
            device = torch.device(dvc)
            return device
        except:
            print(f'Device {dvc} is not available. Defaulting to CPU.')
            return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
    
if __name__ == '__main__':
    sequence = "A group of penguins standing in front of an igloo, laughing until they're completely exhausted."
    model = 't5'
    device = check_device('cpu')
    if model == 't5':
        translate_sequence_from_t5(sequence, device)
    else:
        run_id = "multi30k-small"
        translate_sequence_from_checkpoint(run_id, sequence, device)