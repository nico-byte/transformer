from typing import List, Dict, Tuple
from utils.logger import get_logger
import abc
import random
from src.t5_inference import mt_batch_inference
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.datasets import Multi30k
from datasets import load_dataset
from utils.config import DataLoaderConfig, TokenizerConfig, SharedConfig
from tokenizer import wordpiece_tokenizer
from tokenizer import unigram_tokenizer

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()


class BaseDataLoader(metaclass=abc.ABCMeta):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, shared_config: SharedConfig):
        self.batch_size: int = dl_config.batch_size
        self.num_workers: int = dl_config.num_workers
        self.pin_memory: bool = dl_config.pin_memory
        self.drop_last: bool = dl_config.drop_last
        self.shuffle: bool = dl_config.shuffle
        self.tokenizer: Dict[str] = {}
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

    def collate_fn(self, batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            encoded_src_sample = self.tokenizer['src'].encode(src_sample)
            tensor_src_sample = torch.tensor(encoded_src_sample.ids)
            src_batch.append(tensor_src_sample)

            encoded_tgt_sample = self.tokenizer['tgt'].encode(tgt_sample)
            tensor_tgt_sample = torch.tensor(encoded_tgt_sample.ids)
            tgt_batch.append(tensor_tgt_sample)

        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=self.special_symbols.index("<pad>"))
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.special_symbols.index("<pad>"))

        return src_batch, tgt_batch
    
    def backtranslate_dataset(self, whole_dataset: List[Tuple[str, str]], tgt_dataset: List[str]):
        backtrans_dataset = mt_batch_inference(tgt_dataset, "cuda", 256)
        
        backtrans_dataset_pairs = [[x, y] for x, y in zip(backtrans_dataset, tgt_dataset)]
        
        new_dataset = whole_dataset + backtrans_dataset_pairs
        clean_dataset = self.clean_dataset(new_dataset)
                
        return clean_dataset
    
    def clean_dataset(self, dataset: List[Tuple[str, str]]):
        src_dataset = [x[0] for x in dataset]
        tgt_dataset = [x[1] for x in dataset]
        
        unique_pairs = {}
        for x, y in zip(src_dataset, tgt_dataset):
            if (x and y) and (x not in unique_pairs):
                unique_pairs[x] = y
    
        clean_dataset = [[x, y] for x, y in unique_pairs.items()]
        self.logger.info(f'Cleaned dataset: {len(clean_dataset)}')
    
        return clean_dataset


class IWSLT2017DataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, shared_config: SharedConfig, tokenizer: str="wordpiece"):
        super().__init__(dl_config, tkn_config, shared_config)
        
        self.dataset = load_dataset("iwslt2017", f'iwslt2017-{self.src_language}-{self.tgt_language}', cache_dir='./.data/iwslt2017')
            
        self.build_datasets()
        self.logger.info('Datasets have been loaded.')
                
        src_train_dataset = [x[0] for x in self.train_dataset]
        tgt_train_dataset = [x[1] for x in self.train_dataset]
        
        if tokenizer == "wordpiece":
            self.tokenizer['src'], self.tokenizer['tgt'] = wordpiece_tokenizer.build_tokenizer(name="cased", run_id=shared_config.run_id, src_dataset=src_train_dataset, tgt_dataset=tgt_train_dataset, vocab_size=12280)
            # self.tokenizer['tgt'] = wordpiece_tokenizer.build_tokenizer(name="iwslt-tgt", run_id=shared_config.run_id, dataset=tgt_train_dataset, vocab_size=12280)
        elif tokenizer == "unigram":
            self.tokenizer['src'] = unigram_tokenizer.build_tokenizer(name="iwslt-src", run_id=shared_config.run_id, dataset=src_train_dataset, vocab_size=12280)
            self.tokenizer['tgt'] = unigram_tokenizer.build_tokenizer(name="iwslt-tgt", run_id=shared_config.run_id, dataset=tgt_train_dataset, vocab_size=12280)
        else:
            raise KeyError

        super().build_dataloaders()
        self.logger.info('Dataloaders have been built.')
        
    def build_datasets(self):
        self.train_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["train"]['translation']]
        self.test_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["test"]['translation']]
        self.val_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["validation"]['translation']]


class Multi30kDataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, shared_config: SharedConfig, tokenizer: str="wordpiece"):
        super().__init__(dl_config, tkn_config, shared_config)

        self.build_datasets()
        self.logger.info('Datasets have benn loaded.')
        
        src_train_dataset = [x[0] for x in self.train_dataset]
        tgt_train_dataset = [x[1] for x in self.train_dataset]
        
        if tokenizer == "wordpiece":
            self.tokenizer['src'], self.tokenizer['tgt'] = wordpiece_tokenizer.build_tokenizer(name="cased", run_id=shared_config.run_id, src_dataset=src_train_dataset, tgt_dataset=tgt_train_dataset, vocab_size=1640)
            # self.tokenizer['tgt'] = wordpiece_tokenizer.build_tokenizer(name="multi30k-tgt", run_id=shared_config.run_id, dataset=tgt_train_dataset, vocab_size=1640)
        elif tokenizer == "unigram":
            self.tokenizer['src'] = unigram_tokenizer.build_tokenizer(name="multi30k-src", run_id=shared_config.run_id, dataset=src_train_dataset, vocab_size=1640)
            self.tokenizer['tgt'] = unigram_tokenizer.build_tokenizer(name="multi30k-tgt", run_id=shared_config.run_id, dataset=tgt_train_dataset, vocab_size=1640)
        else:
            raise KeyError

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
        
        # tgt_train_dataset = [x[1] for x in self.train_dataset]
        # self.train_dataset = self.backtranslate_dataset(self.train_dataset, tgt_train_dataset)
        
        self.logger.debug("First Entry train dataset: %s", list(self.train_dataset[0]))
        self.logger.debug("Length train dataset: %f", len(self.train_dataset))
        self.logger.debug("First Entry test dataset: %s", list(self.test_dataset[0]))
        self.logger.debug("Length test dataset: %f", len(self.test_dataset))
        self.logger.debug("First Entry val dataset: %s", list(self.val_dataset[0]))
        self.logger.debug("Length val dataset: %f", len(self.val_dataset))
        
        