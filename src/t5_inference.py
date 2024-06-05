from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer


def get_base_model(device):
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small", cache_dir="./.transformers/")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", cache_dir="./.transformers/")
    
    return tokenizer, model.to(device)


def t5_inference(tokenizer, model, sequence, device):
    model.eval()

    sequence = ["translate English to German: " + sequence]
    input_ids = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def mt_batch_inference(sequences, device, batch_size=1):
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device)
    
    model.eval()
    if not isinstance(sequences, list):
        sequences = [sequences]
        
    outputs = []
    for i in range(0, len(sequences), batch_size):
        print("Augmenting batch", i)
        translations = model.generate(**tokenizer(sequences[i:i+batch_size], return_tensors="pt", padding=True).to(device))
        outputs += tokenizer.batch_decode(translations, skip_special_tokens=True)

    return outputs