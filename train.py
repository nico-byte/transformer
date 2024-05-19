import torch
from torchinfo import summary
import warnings
import yaml
import argparse

import nltk
from data import IWSLT2017DataLoader, Multi30kDataLoader
from transformer import Seq2SeqTransformer
from trainer import Trainer, EarlyStopper
from config import SharedStore, TokenizerConfig, DataLoaderConfig, TransformerConfig, TrainerConfig
from translate import Translate
warnings.filterwarnings("ignore", category=UserWarning)

nltk.download('wordnet', download_dir='./.venv/share/nltk_data')


def parsing_args():
      parser = argparse.ArgumentParser(description='Parsing some important arguments.')
      parser.add_argument('path_to_config', type=str)
      parser.add_argument('--run-id', type=str)
      parser.add_argument('--torch-device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
      parser.add_argument('--resume', default=False, action='store_true')

      return parser.parse_args()

def main(args):
      path_to_config = args.path_to_config
      run_id = args.run_id
      device = args.torch_device
      resume = args.resume
      
      with open(path_to_config) as stream:
            config = yaml.safe_load(stream)
      
      tkn_conf = TokenizerConfig()
      print(tkn_conf.model_dump())


      shared_store = SharedStore(
            token_transform={
                  tkn_conf.src_language: tkn_conf.src_tokenizer,
                  tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
            }
      )
      dl_conf = DataLoaderConfig(**config['dataloader'])
      print(dl_conf.model_dump())

      if dl_conf.dataset == "iwslt2017":
            _ = IWSLT2017DataLoader(dl_conf, tkn_conf, shared_store)
      else:
            _ = Multi30kDataLoader(dl_conf, tkn_conf, shared_store)
            
      print(shared_store.model_dump())
      print(shared_store.vocab_transform)
      
      SRC_VOCAB_SIZE = len(shared_store.vocab_transform[tkn_conf.src_language])
      TGT_VOCAB_SIZE = len(shared_store.vocab_transform[tkn_conf.tgt_language])
            

      model_conf = TransformerConfig(
            **config['transformer'],
            src_vocab_size=SRC_VOCAB_SIZE,
            tgt_vocab_size=TGT_VOCAB_SIZE,
            shared_store=shared_store
      )
      print(model_conf.model_dump())

      transformer = Seq2SeqTransformer(model_conf).to(device)
      translator = Translate(transformer, device, shared_store.special_symbols)

      trainer_conf = TrainerConfig(
            **config['trainer'],
            device=device
      )
      print(trainer_conf.model_dump())
      summary(transformer, [(500, dl_conf.batch_size), (500, dl_conf.batch_size), 
                            (500, 500), (500, 500), 
                            (dl_conf.batch_size, 500), (dl_conf.batch_size, 500)], depth=4)

      early_stopper = EarlyStopper(patience=3, min_delta=0.03)

      trainer = Trainer(transformer, translator, early_stopper, trainer_conf, shared_store, run_id=run_id)

      trainer.train()
      print(f'\nEvaluation: meteor_score - {trainer.evaluate(tgt_language=tkn_conf.tgt_language)}')

      TEST_SEQUENCE = "Ein Mann mit blonden Haar hat ein Haus aus Steinen gebaut ."
      output = translator.translate(TEST_SEQUENCE, src_language=tkn_conf.src_language, 
            tgt_language=tkn_conf.tgt_language, text_transform=shared_store.text_transform, 
            vocab_transform=shared_store.vocab_transform, special_symbols=shared_store.special_symbols)
      
      print(f'Input: {TEST_SEQUENCE}, Output: {output}')
      
if __name__ == '__main__':
      args = parsing_args()
      main(args)
