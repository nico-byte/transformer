import torch
from torchinfo import summary
import warnings

import nltk
from data import IWSLT2017DataLoader, Multi30kDataLoader
from transformer import Seq2SeqTransformer
from trainer import Trainer, EarlyStopper
from config import SharedStore, TokenizerConfig, DataLoaderConfig, TransformerConfig, TrainerConfig
warnings.filterwarnings("ignore", category=UserWarning)

nltk.download('wordnet', download_dir='./.nltk')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tkn_conf = TokenizerConfig()
print(tkn_conf.model_dump())


shared_store = SharedStore(
      token_transform={
            tkn_conf.src_language: tkn_conf.src_tokenizer,
            tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
      }
)

dl_conf = DataLoaderConfig(
      dataset="multi30k",
      batch_size=32,
      num_workers=4,
      pin_memory=True,
      drop_last=False,
      shuffle=True,
)
print(dl_conf.model_dump())
print(shared_store.model_dump())

if dl_conf.dataset == "iwslt2017":
    dataloader = IWSLT2017DataLoader(dl_conf, tkn_conf, shared_store)
else:
      dataloader = Multi30kDataLoader(dl_conf, tkn_conf, shared_store)

SRC_VOCAB_SIZE = len(shared_store.vocab_transform[tkn_conf.src_language])
TGT_VOCAB_SIZE = len(shared_store.vocab_transform[tkn_conf.tgt_language])

model_conf = TransformerConfig(
      num_encoder_layers=4,
      num_decoder_layers=4,
      emb_size=512,
      nhead=8,
      src_vocab_size=SRC_VOCAB_SIZE,
      tgt_vocab_size=TGT_VOCAB_SIZE,
      dim_feedforward=2048,
      dropout=0.1,
      shared_store=shared_store
)
print(model_conf.model_dump())

transformer = Seq2SeqTransformer(model_conf, shared_store).to(DEVICE)

trainer_conf = TrainerConfig(
      learning_rate=0.0001,
      num_epochs=200,
      batch_size=shared_store.dataloaders[0].batch_size,
      tgt_batch_size=1024,
      num_cycles=6,
      device=DEVICE
)
print(trainer_conf.model_dump())

summary(transformer, [(500, dl_conf.batch_size), (500, dl_conf.batch_size)])

early_stopper = EarlyStopper(patience=3, min_delta=0.03)

trainer = Trainer(transformer, early_stopper, trainer_conf, shared_store)

trainer.train()
print(f'\nEvaluation: meteor_score  - {trainer.evaluate(tgt_language=tkn_conf.tgt_language)}')

TEST_SEQUENCE = "Ein Mann mit blonden Haar hat ein Haus aus Steinen gebaut ."

print(f'Input: {TEST_SEQUENCE}, Output: {transformer.translate(TEST_SEQUENCE, src_language=tkn_conf.src_language, tgt_language=tkn_conf.tgt_language)}')

