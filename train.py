from torchinfo import summary
import warnings
import yaml
import argparse
import os
import sys
# import nltk

from src.data import IWSLT2017DataLoader, Multi30kDataLoader
from utils.logger import get_logger
from src.transformer import Seq2SeqTransformer
from src.trainer import Trainer, EarlyStopper
from utils.config import SharedConfig, TokenizerConfig, DataLoaderConfig, TransformerConfig, TrainerConfig
from src.processor import Processor
warnings.filterwarnings("ignore", category=UserWarning)

# if not os.path.exists('./.nltk_data'):
#       nltk.download("wordnet", download_dir='./.nltk')


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
      
      logger = get_logger("Main")
      
      if os.path.exists(f'./models/{run_id}/tokenizer'):
            logger.error('Run ID already exists!')
            sys.exit(1)
      else:
            os.makedirs(f'./models/{run_id}/tokenizer')
      
      with open(path_to_config) as stream:
            config = yaml.safe_load(stream)
      
      tkn_conf = TokenizerConfig()


      shared_conf = SharedConfig(run_id=run_id)
      dl_conf = DataLoaderConfig(**config['dataloader'])

      if dl_conf.dataset == "iwslt2017":
            dataloader = IWSLT2017DataLoader(dl_conf, tkn_conf, shared_conf)
      else:
            dataloader = Multi30kDataLoader(dl_conf, tkn_conf, shared_conf)
            
      train_dataloader, test_dataloader, val_dataloader, tokenizer = dataloader.train_dataloader, dataloader.test_dataloader, dataloader.val_dataloader, dataloader.tokenizer
            
      SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = tokenizer.vocab_size, tokenizer.vocab_size
            

      model_conf = TransformerConfig(
            **config['transformer'],
            src_vocab_size=SRC_VOCAB_SIZE,
            tgt_vocab_size=TGT_VOCAB_SIZE
      )

      transformer = Seq2SeqTransformer(model_conf)
      translator = Processor(transformer, device, shared_conf.special_symbols)

      trainer_conf = TrainerConfig(
            **config['trainer'],
            device=device, 
            batch_size=dl_conf.batch_size
      )
      summary(transformer, [(256, dl_conf.batch_size), (256, dl_conf.batch_size), 
                            (256, 256), (256, 256), 
                            (dl_conf.batch_size, 256), (dl_conf.batch_size, 256)], depth=4)

      early_stopper = EarlyStopper(warmup=17, patience=7, min_delta=0)

      trainer = Trainer(transformer, translator, train_dataloader, test_dataloader, val_dataloader, 
                        tokenizer, early_stopper, trainer_conf, shared_conf, run_id, device)

      trainer.train()
      bleu, rouge = trainer.evaluate()
      print(f'\nEvaluation: bleu_score - {bleu}, rouge_score - {rouge}')

      TEST_SEQUENCE = "The quick brown fox jumped over the lazy dog and then ran away quickly."
      output = translator.translate(TEST_SEQUENCE, tokenizer=tokenizer, special_symbols=shared_conf.special_symbols)
      
      print(f'Input: {TEST_SEQUENCE}, Output: {output}')
      
if __name__ == '__main__':
      args = parsing_args()
      main(args)
