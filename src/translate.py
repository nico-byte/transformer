import torch
import tokenizers
from src.processor import Processor


def translate_sequence_from_checkpoint(checkpoint, tokenizer, sequence, device):
    checkpoint = torch.jit.load(checkpoint, map_location=device)
    
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer)
        
    translator = Processor(checkpoint, tokenizer, device)
    
    output = translator.translate(sequence)
    
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
