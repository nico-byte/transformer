import torch
import tokenizers
from src.processor import Processor
from utils.config import SharedConfig


def translate_sequence_from_checkpoint(checkpoint, tokenizer, sequence, device):
    """
    Translate a sequence using a model checkpoint.

    Args:
        checkpoint (str): Path to the model checkpoint.
        tokenizer (str): Path to the tokenizer.
        sequence (str): Input sequence to translate.
        device: Device to run the model on.

    Returns:
        str: Translated sequence.
    """

    checkpoint = torch.jit.load(checkpoint, map_location=device)

    shared_config = SharedConfig()
    
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer)
    special_symbols = shared_config.special_symbols
        
    translator = Processor(checkpoint, device, special_symbols)
    
    output = translator.translate(sequence, tokenizer, special_symbols=special_symbols)
    
    return output    
    

def check_device(dvc=None):
    """
    Check and return the appropriate torch device.

    Args:
        dvc: Device string. Defaults to None.

    Returns:
        torch.device: The selected device.
    """

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
