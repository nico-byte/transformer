from torchinfo import summary
import warnings
import yaml
import argparse

import nltk
from data import IWSLT2017DataLoader, Multi30kDataLoader
from transformer import Seq2SeqTransformer
from trainer import Trainer, EarlyStopper
from config import SharedConfig, TokenizerConfig, DataLoaderConfig, TransformerConfig, TrainerConfig
from processor import Processor
warnings.filterwarnings("ignore", category=UserWarning)

nltk.download('wordnet', download_dir='./.venv/share/nltk_data')


def parsing_args():
      parser = argparse.ArgumentParser(description='Parsing some important arguments.')
      parser.add_argument('path_to_config', type=str)
      parser.add_argument('--run-id', type=str)
      parser.add_argument('--torch-device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])

      return parser.parse_args()

def main(args):
      path_to_config = args.path_to_config
      run_id = args.run_id
      device = args.torch_device
      
      with open(path_to_config) as stream:
            config = yaml.safe_load(stream)
      
      tkn_conf = TokenizerConfig()
      
      tokenizer = {
            tkn_conf.src_language: tkn_conf.src_tokenizer,
            tkn_conf.tgt_language: tkn_conf.tgt_tokenizer
      }


      shared_conf = SharedConfig()
      dl_conf = DataLoaderConfig(**config['dataloader'])

      if dl_conf.dataset == "iwslt2017":
            dataloader = IWSLT2017DataLoader(dl_conf, tokenizer, tkn_conf, shared_conf)
      else:
            dataloader = Multi30kDataLoader(dl_conf, tokenizer, tkn_conf, shared_conf)
            
      vocab_transform, text_transform = dataloader.vocab_transform, dataloader.text_transform
      train_dataloader, test_dataloader, val_dataloader = dataloader.train_dataloader, dataloader.test_dataloader, dataloader.val_dataloader
            
      SRC_VOCAB_SIZE = len(vocab_transform[tkn_conf.src_language].index2word)
      TGT_VOCAB_SIZE = len(vocab_transform[tkn_conf.tgt_language].index2word)
            

      model_conf = TransformerConfig(
            **config['transformer'],
            src_vocab_size=SRC_VOCAB_SIZE,
            tgt_vocab_size=TGT_VOCAB_SIZE
      )

      transformer = Seq2SeqTransformer(model_conf)
      translator = Processor(transformer, device, shared_conf.special_symbols)

      trainer_conf = TrainerConfig(
            **config['trainer'],
            device=device
      )
      summary(transformer, [(256, dl_conf.batch_size), (256, dl_conf.batch_size), 
                            (256, 256), (256, 256), 
                            (dl_conf.batch_size, 256), (dl_conf.batch_size, 256)], depth=4)

      early_stopper = EarlyStopper(patience=3, min_delta=0)

      trainer = Trainer(transformer, translator, train_dataloader, test_dataloader, val_dataloader, 
                        vocab_transform, early_stopper, trainer_conf, shared_conf, run_id, device)

      trainer.train()
      print(f'\nEvaluation: meteor_score - {trainer.evaluate(tgt_language=tkn_conf.tgt_language)}')

      TEST_SEQUENCE = "Eine Gruppe Pinguine steht vor einem Iglu und lacht sich tot ."
      output = translator.translate(TEST_SEQUENCE, src_language=tkn_conf.src_language, 
            tgt_language=tkn_conf.tgt_language, text_transform=text_transform, 
            vocab_transform=vocab_transform, special_symbols=shared_conf.special_symbols)
      
      print(f'Input: {TEST_SEQUENCE}, Output: {output}')
      
if __name__ == '__main__':
      args = parsing_args()
      main(args)
