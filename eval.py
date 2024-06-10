import warnings
import argparse

from src.data import IWSLT2017DataLoader, Multi30kDataLoader
from src.trainer import Trainer
from utils.config import SharedConfig, TokenizerConfig, DataLoaderConfig
from src.processor import Processor
warnings.filterwarnings("ignore", category=UserWarning)


def parsing_args():
      parser = argparse.ArgumentParser(description='Parsing some important arguments.')
      parser.add_argument('path_to_checkpoint', type=str)
      parser.add_argument('path_to_tokenizer', type=str)
      parser.add_argument('--torch-device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])

      return parser.parse_args()

def main(args):
      path_to_checkpoint = args.path_to_checkpoint
      path_to_tokenizer = args.path_to_tokenizer
      device = args.torch_device
            
      tkn_conf = TokenizerConfig()


      shared_conf = SharedConfig()
      dl_conf = DataLoaderConfig()

      if dl_conf.dataset == "iwslt2017":
            dataloader = IWSLT2017DataLoader.build_with_tokenizer(dl_conf, tkn_conf, shared_conf, path_to_tokenizer)
      else:
            dataloader = Multi30kDataLoader(dl_conf, tkn_conf, shared_conf)
            
      val_dataloader = dataloader.val_dataloader
            
      translator = Processor.from_checkpoint(model_checkpoint=path_to_checkpoint, 
                                             tokenizer=path_to_tokenizer, 
                                             device=device)


      bleu, rouge = Trainer.evaluate_checkpoint(checkpoint_path=path_to_checkpoint, 
                                                tokenizer_path=path_to_tokenizer, 
                                                val_dataloader=val_dataloader, 
                                                translator=translator,
                                                device=device)

      print(f'\nEvaluation: bleu_score - {bleu}, rouge_score - {rouge}')

      TEST_SEQUENCE = "The quick brown fox jumped over the lazy dog and then ran away quickly."
      output = translator.translate(TEST_SEQUENCE)
      
      print(f'Input: {TEST_SEQUENCE}, Output: {output}')
      
if __name__ == '__main__':
      args = parsing_args()
      main(args)