from torchinfo import summary
import warnings
import yaml
import argparse

import nltk
from src.data import IWSLT2017DataLoader, Multi30kDataLoader
from src.transformer import Seq2SeqTransformer
from tokenizers import Tokenizer
from src.trainer import Trainer, EarlyStopper
from utils.config import SharedConfig, TokenizerConfig, DataLoaderConfig, TransformerConfig, TrainerConfig
from src.processor import Processor
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
      src_lang, tgt_lang = tkn_conf.src_language, tkn_conf.tgt_language
      
      tokenizer = {
            src_lang: Tokenizer.from_file("./tokenizer/wordpiece/cased-en-multi.json"),
            tgt_lang: Tokenizer.from_file("./tokenizer/wordpiece/cased-de-multi.json")
      }


      shared_conf = SharedConfig()
      dl_conf = DataLoaderConfig(**config['dataloader'])

      if dl_conf.dataset == "iwslt2017":
            dataloader = IWSLT2017DataLoader(dl_conf, tokenizer, tkn_conf, shared_conf)
      else:
            dataloader = Multi30kDataLoader(dl_conf, tokenizer, tkn_conf, shared_conf)
            
      train_dataloader, test_dataloader, val_dataloader = dataloader.train_dataloader, dataloader.test_dataloader, dataloader.val_dataloader
            
      SRC_VOCAB_SIZE = tokenizer[src_lang].get_vocab_size()
      TGT_VOCAB_SIZE = tokenizer[tgt_lang].get_vocab_size()
            

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

      early_stopper = EarlyStopper(warmup=trainer_conf.warmup_epochs, patience=3, min_delta=0)

      trainer = Trainer(transformer, translator, train_dataloader, test_dataloader, val_dataloader, 
                        tokenizer[tgt_lang], early_stopper, trainer_conf, shared_conf, run_id, device)

      trainer.train()
      print(f'\nEvaluation: meteor_score - {trainer.evaluate()}')

      TEST_SEQUENCE = "The quick brown fox jumped over the lazy dog and then ran away quickly."
      output = translator.translate(TEST_SEQUENCE, src_language=tkn_conf.src_language, 
            tgt_language=tkn_conf.tgt_language, tokenizer=tokenizer, 
            special_symbols=shared_conf.special_symbols)
      
      print(f'Input: {TEST_SEQUENCE}, Output: {output}')
      
if __name__ == '__main__':
      args = parsing_args()
      main(args)
