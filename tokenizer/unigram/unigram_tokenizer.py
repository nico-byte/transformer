from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import os


de_tokenizer = Tokenizer(models.Unigram())
de_tokenizer.normalizer = normalizers.NFKC()
de_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
de_tokenizer.decoder = decoders.ByteLevel()
de_tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[("<bos>", 1), ("<eos>", 2)]
)

en_tokenizer = Tokenizer(models.Unigram())
en_tokenizer.normalizer = normalizers.NFKC()
en_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
en_tokenizer.decoder = decoders.ByteLevel()
en_tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[("<bos>", 1), ("<eos>", 2)]
)

trainer = trainers.UnigramTrainer(
    vocab_size=12288,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
)

dataset = load_dataset("iwslt2017", 'iwslt2017-de-en', cache_dir='./.data/iwslt2017')
dataset = [(d["de"], d["en"]) for d in dataset["train"]['translation']]

de_dataset = [x[0] for x in dataset]
en_dataset = [x[1] for x in dataset]

print(de_dataset[0], en_dataset[0])

def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]
                
de_tokenizer.train_from_iterator(batch_iterator(de_dataset), trainer=trainer, length=len(de_dataset))
en_tokenizer.train_from_iterator(batch_iterator(en_dataset), trainer=trainer, length=len(en_dataset))

if not os.path.exists('./tokenizer/unigram'):
    os.makedirs('./tokenizer/unigram')
        
de_tokenizer.save("./tokenizer/unigram/cased-de.json")
en_tokenizer.save("./tokenizer/unigram/cased-en.json")

de_tokenizer = Tokenizer.from_file("./tokenizer/unigram/cased-de.json")
en_tokenizer = Tokenizer.from_file("./tokenizer/unigram/cased-en.json")

de_input = de_tokenizer.encode("Der schnelle braune Fuchs sprang über den faulen Hund und lief dann schnell weg.")
en_input = en_tokenizer.encode("The quick brown fox jumped over the lazy dog and then ran away quickly.")

print(de_input.ids, de_input.tokens)
print(en_input.ids, en_input.tokens)

de_output = de_tokenizer.decode(de_input.ids)
en_output = en_tokenizer.decode(en_input.ids)

print(de_output, en_output)
