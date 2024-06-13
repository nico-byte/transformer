from torchinfo import summary
import warnings
import yaml
import argparse
import os
import sys

from src.data import IWSLT2017DataLoader, Multi30kDataLoader
from utils.logger import get_logger
from src.transformer import Seq2SeqTransformer
from src.trainer import Trainer, EarlyStopper
from utils.config import SharedConfig, DataLoaderConfig, TransformerConfig, TrainerConfig
from src.processor import Processor
warnings.filterwarnings("ignore", category=UserWarning)


def parsing_args():
      """
      Parse command line arguments for the training and evaluation pipeline.

      The function creates an argument parser, adds the necessary arguments,    
      and parses them from the command line input. The expected arguments are:
      - path_to_config (str): The path to the configuration YAML file.
      - run-id (str, optional): A unique identifier for the training run.
      - torch-device (str, optional): The device to run the model on, with
        possible choices being 'cpu', 'cuda', 'cuda:0', or 'cuda:1'. Defaults to 'cpu'.

      """

      parser = argparse.ArgumentParser(description='Parsing some important arguments.')
      parser.add_argument('path_to_config', type=str)
      parser.add_argument('--run-id', type=str)
      parser.add_argument('--torch-device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])

      return parser.parse_args()

def main(args):
      """
      
      Main function to execute the training and evaluation pipeline for a Seq2Seq Transformer model.

      The function performs the following steps:
      1. Initializes the logger.
      2. Checks if the specified run ID already exists and creates a directory for it if it doesn't.
      3. Loads the configuration from the provided YAML file.
      4. Initializes tokenizer, shared, and dataloader configurations.
      5. Loads the appropriate dataset based on the configuration.
      6. Initializes the Seq2Seq Transformer model and processor.
      7. Sets up the trainer configuration and model summary.
      8. Initializes early stopping criteria and the trainer.
      9. Trains the model and evaluates its performance.
      10. Demonstrates the model's translation capability with a test sequence.

      Args:
            args: Parsed command line arguments containing:
                  - path_to_config (str): Path to the configuration YAML file.
                  - run_id (str): Unique run identifier.
                  - torch_device (str): Device to run the model on (e.g., 'cpu', 'cuda', etc.).
      Raises:
            SystemExit: If the specified run ID already exists.

      """
      path_to_config: str = args.path_to_config
      run_id: str = args.run_id
      device: str = args.torch_device
      
      logger = get_logger("Main")
      
      if os.path.exists(f'./models/{run_id}/metrics'):
            logger.error('Run ID already exists!')
            sys.exit(1)
      else:
            os.makedirs(f'./models/{run_id}/metrics')
      
      with open(path_to_config) as stream:
            config = yaml.safe_load(stream)

      shared_conf = SharedConfig(run_id=run_id)
      dl_conf = DataLoaderConfig(**config['dataloader'])

      if dl_conf.dataset == "iwslt2017":
            dataloader = IWSLT2017DataLoader.new_instance(dl_conf, shared_conf)
      else:
            dataloader = Multi30kDataLoader.new_instance(dl_conf, shared_conf)
            
      train_dataloader, test_dataloader, val_dataloader, tokenizer = dataloader.train_dataloader, dataloader.test_dataloader, dataloader.val_dataloader, dataloader.tokenizer
            
      SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = tokenizer.get_vocab_size(), tokenizer.get_vocab_size()
            

      model_conf = TransformerConfig(
            **config['transformer'],
            src_vocab_size=SRC_VOCAB_SIZE,
            tgt_vocab_size=TGT_VOCAB_SIZE
      )

      transformer = Seq2SeqTransformer(model_conf)
      translator = Processor.from_instance(transformer, tokenizer, device)

      trainer_conf = TrainerConfig(
            **config['trainer'],
            device=device, 
            batch_size=dl_conf.batch_size
      )
      summary(transformer, [(256, dl_conf.batch_size), (256, dl_conf.batch_size), 
                            (256, 256), (256, 256), 
                            (dl_conf.batch_size, 256), (dl_conf.batch_size, 256)], depth=3)

      early_stopper = EarlyStopper(warmup=17, patience=7, min_delta=0)

      trainer = Trainer.new_instance(transformer, translator, train_dataloader, test_dataloader, val_dataloader, 
                        tokenizer, early_stopper, trainer_conf, device, run_id)

      trainer.train()
      bleu, rouge = trainer.evaluate()
      print(f'\nEvaluation: bleu_score - {bleu}, rouge_score - {rouge}')

      TEST_SEQUENCE = "The quick brown fox jumped over the lazy dog and then ran away quickly."
      output: str = translator.translate(TEST_SEQUENCE)
      
      print(f'Input: {TEST_SEQUENCE}, Output: {output}')
      
if __name__ == '__main__':
      args = parsing_args()
      main(args)
