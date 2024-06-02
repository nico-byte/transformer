from typing import List, Dict, Tuple, Any
from logger import get_logger
import abc
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.datasets import Multi30k
from datasets import load_dataset
from config import DataLoaderConfig, TokenizerConfig, SharedConfig

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()


class BaseDataLoader(metaclass=abc.ABCMeta):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, tokenizer, shared_config: SharedConfig):
        self.batch_size: int = dl_config.batch_size
        self.num_workers: int = dl_config.num_workers
        self.pin_memory: bool = dl_config.pin_memory
        self.drop_last: bool = dl_config.drop_last
        self.shuffle: bool = dl_config.shuffle
        self.tokenizer: Dict[str] = tokenizer
        self.src_language: str = tkn_config.src_language
        self.tgt_language: str = tkn_config.tgt_language
        self.special_symbols: List[str] = shared_config.special_symbols
        
        self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
        self.train_dataloader, self.test_dataloader, self.val_dataloader = None, None, None
        
        self.logger = get_logger('DataLoader')

    @abc.abstractmethod
    def build_datasets(self):
        pass

    def build_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           collate_fn=self.collate_fn, 
                                           shuffle=self.shuffle, 
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=self.drop_last)
        self.test_dataloader = DataLoader(self.test_dataset, 
                                          batch_size=self.batch_size, 
                                          collate_fn=self.collate_fn, 
                                          shuffle=self.shuffle, 
                                          num_workers=self.num_workers,
                                          pin_memory=self.pin_memory,
                                          drop_last=self.drop_last)
        self.val_dataloader = DataLoader(self.val_dataset, 
                                         batch_size=self.batch_size, 
                                         collate_fn=self.collate_fn, 
                                         shuffle=self.shuffle, 
                                         num_workers=self.num_workers,
                                         pin_memory=self.pin_memory,
                                         drop_last=self.drop_last)

    def collate_fn(self, batch: List[str]) -> Tuple[List[str], List[str]]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_sample = self.tokenizer[self.src_language].encode(src_sample.rstrip("\n")).ids
            src_sample = torch.tensor(src_sample)
            src_batch.append(src_sample)
            
            tgt_sample = self.tokenizer[self.tgt_language].encode(tgt_sample.rstrip("\n")).ids
            tgt_sample = torch.tensor(tgt_sample)
            tgt_batch.append(tgt_sample)

        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=self.special_symbols.index('<pad>'))
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.special_symbols.index('<pad>'))

        return src_batch, tgt_batch


class IWSLT2017DataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, tokenizer: Any, shared_config: SharedConfig):
        super().__init__(dl_config, tokenizer, tkn_config, shared_config)
        
        self.dataset = load_dataset("iwslt2017", f'iwslt2017-{self.src_language}-{self.tgt_language}', cache_dir='./.data/iwslt2017')
            
        self.build_datasets()
        self.logger.info('Datasets have benn loaded.')

        super().build_dataloaders()
        self.logger.info('Dataloaders have been built.')

    def build_datasets(self):
        self.train_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["train"]['translation']]
        self.test_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["test"]['translation']]
        self.val_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["validation"]['translation']]


class Multi30kDataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, tokenizer: Any, shared_config: SharedConfig):
        super().__init__(dl_config, tokenizer, tkn_config, shared_config)

        self.build_datasets()
        self.logger.info('Datasets have benn loaded.')

        super().build_dataloaders()
        self.logger.info('Dataloaders have been built.')

    def build_datasets(self):
        self.train_dataset: List[str, str] = list(Multi30k(root='./.data/multi30k', split='train',
                                      language_pair=(self.src_language, self.tgt_language)))
        
        self.test_dataset: List[str, str] = list(Multi30k(root='./.data/multi30k',  split='valid',
                                    language_pair=(self.src_language, self.tgt_language)))
        
        total_entries = len(self.train_dataset)
        num_test_entries = int(total_entries * 0.05)
        val_indices = random.sample(range(total_entries), num_test_entries)
        
        self.val_dataset: List[str, str] = [self.train_dataset[i] for i in val_indices]
        self.train_dataset = [entry for i, entry in enumerate(self.train_dataset) if i not in val_indices]
        