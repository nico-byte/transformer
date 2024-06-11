from transformers import MarianTokenizer

# Load the tokenizer from the pretrained model
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", cache_dir="./.transformers")

# Define source and target texts for translation
src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional

# Tokenize the inputs
inputs = tokenizer(src_texts, text_target=tgt_texts)

print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))

# Print special tokens and their IDs
print(tokenizer.unk_token, tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token)
print(tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)

# Save the tokenizer
tokenizer.save_pretrained("./tokenizer.json")

# Reload the tokenizer from the saved file
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", cache_dir="./.transformers")

# Load the tokenizer from the saved file
tokenizer = MarianTokenizer("./tokenizer.json/source.spm", "./tokenizer.json/target.spm", "./tokenizer.json/vocab.json", "en", "de")

src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional

# Tokenize the inputs again
inputs = tokenizer(src_texts, text_target=tgt_texts)

print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))

# Print special tokens and their IDs again
print(tokenizer.unk_token, tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token)
print(tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)

