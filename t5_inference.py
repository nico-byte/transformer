from transformers import T5Tokenizer, T5ForConditionalGeneration


def get_base_model():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small", cache_dir="./.transformers/")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", cache_dir="./.transformers/")
    
    return tokenizer, model


def t5_inference(tokenizer, model, sequence, device):
    model.to(device)
    model.eval()

    sequence = ["translate English to German: " + sequence]
    input_ids = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)