from transformers import MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", cache_dir="./.transformers")
src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
inputs = tokenizer(src_texts, text_target=tgt_texts)

print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))
print(tokenizer.unk_token, tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token)
print(tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)

tokenizer.save_pretrained("./tokenizer.json")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", cache_dir="./.transformers")

tokenizer = MarianTokenizer.from_pretrained("./tokenizer.json/")

src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
inputs = tokenizer(src_texts, text_target=tgt_texts)

print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))
print(tokenizer.unk_token, tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token)
print(tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)

