from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer
from torch.cuda.amp import autocast


def get_base_model(device):
    """
    Load a base T5 model and tokenizer from the 'google-t5/t5-small' checkpoint.

    Args:
        device: The device to run the model on.

    Returns:
        Tuple[T5Tokenizer, T5ForConditionalGeneration]: A tuple containing the tokenizer and the model, both loaded and moved to the specified device.
    """

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small", cache_dir="./.transformers/")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", cache_dir="./.transformers/")
    
    return tokenizer, model.to(device)


def t5_inference(tokenizer, model, sequence, device):
    """
    Perform inference using a T5 model for translation from English to German.

    Args:
        tokenizer: The tokenizer for the T5 model.
        model: The T5 model for translation.
        sequence (str): The input sequence to translate from English to German.
        device: The device to run the model on.

    Returns:
        str: The translated sequence from English to German.
    """

    model.eval()

    sequence = ["translate English to German: " + sequence]
    input_ids = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def mt_batch_inference(sequences, device, batch_size, logger):
    """
    Perform batch-based machine translation inference using a pre-trained Marian MT model.
    
    Args:
        sequences (list[str]): A list of input sequences to be translated.
        device (torch.device): The device to run the model on.
        batch_size (int): The batch size to use for inference.
        logger (logging.Logger): A logger object to log progress.
    
    Returns:
        list[str]: The translated sequences.
    """
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device)
    
    model.eval()
    if not isinstance(sequences, list):
        sequences = [sequences]
        
    outputs = []
    with autocast():
        for i in range(0, len(sequences), batch_size):
            logger.info(f"Backtranslating batch: {(int(i/batch_size))+1}/{(len(sequences)//batch_size)+1}")
        
            with autocast():
                translations = model.generate(**tokenizer(sequences[i:i+batch_size], return_tensors="pt", padding=True).to(device))
        
            outputs += tokenizer.batch_decode(translations, skip_special_tokens=True)

    return outputs