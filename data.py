from typing import List
import abc
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from config import DataLoaderConfig, TokenizerConfig, SharedStore

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()


class BaseDataLoader(metaclass=abc.ABCMeta):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, shared_store: SharedStore):
        self.dl_config = dl_config
        self.tkn_config = tkn_config
        self.shared_store = shared_store
        
        self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
        self.dataloaders = []

    @abc.abstractmethod
    def build_datasets(self):
        pass

    def build_dataloaders(self):
        self.shared_store.dataloaders.append(DataLoader(self.train_dataset, 
                                                        batch_size=self.dl_config.batch_size, 
                                                        collate_fn=self.collate_fn, 
                                                        shuffle=self.dl_config.shuffle, 
                                                        num_workers=self.dl_config.num_workers,
                                                        pin_memory=self.dl_config.pin_memory,
                                                        drop_last=self.dl_config.drop_last))
        self.shared_store.dataloaders.append(DataLoader(self.test_dataset, 
                                                        batch_size=self.dl_config.batch_size, 
                                                        collate_fn=self.collate_fn, 
                                                        shuffle=self.dl_config.shuffle, 
                                                        num_workers=self.dl_config.num_workers,
                                                        pin_memory=self.dl_config.pin_memory,
                                                        drop_last=self.dl_config.drop_last))
        self.shared_store.dataloaders.append(DataLoader(self.val_dataset, 
                                                        batch_size=self.dl_config.batch_size, 
                                                        collate_fn=self.collate_fn, 
                                                        shuffle=self.dl_config.shuffle, 
                                                        num_workers=self.dl_config.num_workers,
                                                        pin_memory=self.dl_config.pin_memory,
                                                        drop_last=self.dl_config.drop_last))

    def build_vocab(self):
        vocab_transform = {}
        gibberish_tokens = ['sadhads', 'suhd', 'ksjhd', 'skdsd', 'sjdic', 'sajn']
        for ln in [self.tkn_config.src_language, self.tkn_config.tgt_language]:
            train_iter = self.train_dataset
            vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                            min_freq=1,
                                                            specials=self.shared_store.special_symbols,
                                                            special_first=True)
            # pad vocab to be divisible by 8
            rest_of_8 = 8 - (len(vocab_transform[ln]) % 8)
            for token in gibberish_tokens[:rest_of_8]:
                vocab_transform[ln].append_token(token)
            
            vocab_transform[ln].set_default_index(self.shared_store.special_symbols.index('<unk>'))
        return vocab_transform

    def yield_tokens(self, data_iter: list, language: str):
        language_index = {self.tkn_config.src_language: 0, self.tkn_config.tgt_language: 1}
        for data_sample in data_iter:
            yield self.shared_store.token_transform[language](data_sample[language_index[language]])

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.shared_store.text_transform[self.tkn_config.src_language](src_sample.rstrip("\n")))
            tgt_batch.append(self.shared_store.text_transform[self.tkn_config.tgt_language](tgt_sample.rstrip("\n")))

        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=self.shared_store.special_symbols.index('<pad>'))
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.shared_store.special_symbols.index('<pad>'))

        return src_batch, tgt_batch

    def data_iter(self, iterator):
        for data in iterator:
            yield data


class IWSLT2017DataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, shared_store: SharedStore):
        super().__init__(dl_config, tkn_config, shared_store)
        self.dl_config = dl_config
        self.tkn_config = tkn_config
        
        self.dataset = load_dataset("iwslt2017", f'iwslt2017-{self.tkn_config.src_language}-{self.tkn_config.tgt_language}', cache_dir='./.data/iwslt2017')
        
        self.build_datasets()

        self.shared_store.vocab_transform = super().build_vocab()
        self.shared_store.text_transform = create_text_transform(tkn_config.src_language, tkn_config.tgt_language, 
                                                                 self.shared_store.token_transform, self.shared_store.vocab_transform)

        super().build_dataloaders()

    def build_datasets(self):
        self.train_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["train"]['translation']]
        self.test_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["test"]['translation']]
        self.val_dataset: List[str, str] = [(d["de"], d["en"]) for d in self.dataset["validation"]['translation']]

        print(f'Sample from trainset: {self.train_dataset[0]}\n'
              f'Sample from testset: {self.test_dataset[0]}\n'
              f'Sample from valset: {self.val_dataset[1]}\n')


class Multi30kDataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, shared_store: SharedStore):
        super().__init__(dl_config, tkn_config, shared_store)
        self.dl_config = dl_config
        self.tkn_config = tkn_config

        self.build_datasets()
        
        self.shared_store.vocab_transform = super().build_vocab()
        self.shared_store.text_transform = create_text_transform(tkn_config.src_language, tkn_config.tgt_language, 
                                                                 self.shared_store.token_transform, self.shared_store.vocab_transform)

        super().build_dataloaders()

    def build_datasets(self):
        self.train_dataset: List[str, str] = list(Multi30k(root='./.data/multi30k', split='train',
                                      language_pair=(self.tkn_config.src_language, self.tkn_config.tgt_language)))
        
        self.test_dataset: List[str, str] = list(Multi30k(root='./.data/multi30k',  split='valid',
                                    language_pair=(self.tkn_config.src_language, self.tkn_config.tgt_language)))
        
        total_entries = len(self.train_dataset)
        num_test_entries = int(total_entries * 0.05)
        val_indices = random.sample(range(total_entries), num_test_entries)
        
        self.val_dataset: List[str, str] = [self.train_dataset[i] for i in val_indices]
        self.train_dataset = [entry for i, entry in enumerate(self.train_dataset) if i not in val_indices]

        print(f'Sample from trainset: {self.train_dataset[0]}\n'
              f'Sample from valset: {self.val_dataset[0]}\n')
        

def create_text_transform(src_lang, tgt_lang, token_transform, vocab_transform):
    text_transform = {}
    for ln in [src_lang, tgt_lang]:
        text_transform[ln] = sequential_transforms(token_transform[ln],
                                                    vocab_transform[ln],
                                                    tensor_transform)
    print(text_transform)
    return text_transform

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    special_symbols: List[str] = ['<unk>', '<bos>', '<eos>', '<pad>']
    return torch.cat((torch.tensor([special_symbols.index('<bos>')]),
                      torch.tensor(token_ids),
                      torch.tensor([special_symbols.index('<eos>')])))
    
def load_vocab(run_id):
        vocab_file_path = f'./results/{run_id}/vocab.pth'
        
        return torch.load(vocab_file_path)
        