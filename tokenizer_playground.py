from transformers import AutoTokenizer
from tokenizer.wordpiece_tokenizer import build_tokenizer
from datasets import load_dataset
import random

tokenizer = AutoTokenizer.from_pretrained(
    "google-t5/t5-base", cache_dir="./.transformers"
)

dataset = load_dataset("iwslt2017", "iwslt2017-de-en", cache_dir="./.data/iwslt2017")
dataset = [(d["de"], d["en"]) for d in dataset["train"]["translation"]]

de_dataset = [x[0] for x in dataset]
en_dataset = [x[1] for x in dataset]
whole_dataset = de_dataset + en_dataset


def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


random.shuffle(whole_dataset)

tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(whole_dataset), 6560, len(whole_dataset)
)

src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
inputs = tokenizer(src_texts, text_target=tgt_texts)

print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))
print(tokenizer.decode(inputs["labels"][0]))
print(tokenizer.vocab_size)
print(
    tokenizer.unk_token, tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token
)
print(
    tokenizer.unk_token_id,
    tokenizer.pad_token_id,
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
)

tokenizer.save_pretrained("./tokenizer.json")

tokenizer = AutoTokenizer.from_pretrained("./tokenizer.json/")

src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
inputs = tokenizer(src_texts, text_target=tgt_texts)

print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))
print(
    tokenizer.unk_token, tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token
)
print(
    tokenizer.unk_token_id,
    tokenizer.pad_token_id,
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
)

print(
    tokenizer.tokenize(
        "Tom asked his teacher for advice in generally bad situations he had scared his head."
    )
)
print(
    tokenizer.tokenize(
        "Tom bat seinen Lehrer um Rat in Situationen, in denen er seinen Kopf geschützt hatte."
    )
)

src_wordpiece_tokenizer = build_tokenizer(
    "wp_tokenizer_test", en_dataset, en_dataset, 6540
)

src_test = src_wordpiece_tokenizer.encode(
    "Tom asked his teacher for advice in generally bad situations he had scared his head."
)
tgt_test = src_wordpiece_tokenizer.encode(
    "Tom bat seinen Lehrer um Rat in Situationen, in denen er seinen Kopf geschützt hatte."
)

print(src_test.tokens)
print(tgt_test.tokens)
