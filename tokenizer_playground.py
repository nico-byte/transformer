from config import TokenizerConfig, DataLoaderConfig, SharedConfig
from data import IWSLT2017DataLoader
import yaml
import os

with open("./configs/iwslt2017-small.yaml") as stream:
    config = yaml.safe_load(stream)
      
tkn_conf = TokenizerConfig()
      
tokenizer = {
    tkn_conf.src_language: tkn_conf.src_tokenizer,
    tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
}


shared_conf = SharedConfig()
dl_conf = DataLoaderConfig(**config['dataloader'])

iwslt_dataloader = IWSLT2017DataLoader(dl_conf, tokenizer, tkn_conf, shared_conf)

print(iwslt_dataloader.train_dataset[0])

plain_train_dataset = [x for sublist in [[x[0] for x in iwslt_dataloader.train_dataset], [x[1] for x in iwslt_dataloader.train_dataset]] for x in sublist]
print(len(plain_train_dataset))
print(plain_train_dataset[0])


vocab_transform = iwslt_dataloader.vocab_transform
            
SRC_VOCAB_SIZE = len(vocab_transform[tkn_conf.src_language].index2word)
TGT_VOCAB_SIZE = len(vocab_transform[tkn_conf.tgt_language].index2word)

print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=16384, special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"])
tokenizer.train_from_iterator(plain_train_dataset, trainer)

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[("<bos>", 1), ("<eos>", 2)]
)

from tokenizers import decoders
tokenizer.decoder = decoders.WordPiece()

if not os.path.exists('./tokenizers/'):
    os.makedirs('./tokenizers/')
    
tokenizer.save("./tokenizers/cased-en-de.json")

tokenizer = Tokenizer.from_file("./tokenizers/cased-en-de.json")

src_input = tokenizer.encode("Hallo, ich bin ein Mensch, der sich nich traut auch mal auf den Tisch zu hauen da ich Angst habe den Tisch kaputt zu machen.")
tgt_input = tokenizer.encode("Hello, I am a human being who does not like to smash on the table because i am anxious about the table being broken.")

print(src_input.ids, src_input.tokens)
print(tgt_input.ids, tgt_input.tokens)

src_output = tokenizer.decode(src_input.ids)
tgt_output = tokenizer.decode(tgt_input.ids)

print(src_output, tgt_output)
print(tokenizer.get_vocab_size())