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


class Vocabulary:
    def __init__(self, name: str):
        UNK_TOKEN = 0
        BOS_TOKEN = 1
        EOS_TOKEN = 2
        PAD_TOKEN = 3
    
        self.name: str = name
        self.word2index: Dict[str, int] = {"<unk>": UNK_TOKEN, "<bos>": BOS_TOKEN, "<eos>": EOS_TOKEN, "<pad>": PAD_TOKEN}
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {UNK_TOKEN: "<unk>", BOS_TOKEN: "<bos>", EOS_TOKEN: "<eos>", PAD_TOKEN: "<pad>"}
        self.num_words: int = 4
            
    def add_words(self, words: List[str]) -> List[int]:
        if not isinstance(words, List): words = [words]
        for word in words:
            if word not in self.word2index:
                # First entry of word into vocabulary
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                # Word exists; increase word count
                self.word2count[word] += 1
    
    def to_words(self, indices: List[int]) -> List[str]:
        if not isinstance(indices, List): indices = [indices]
        words = []
        for index in indices:
            if index not in self.index2word:
                words.append(self.index2word[0])
                continue
            words.append(self.index2word[index])
        return words
    
    def to_index(self, words: List[str]) -> List[int]:
        if not isinstance(words, List): words = [words]
        indices = []
        for word in words:
            if word not in self.word2index:
                indices.append(int(self.word2index['<unk>']))
                continue
            indices.append(int(self.word2index[word]))
        return indices


class BaseDataLoader(metaclass=abc.ABCMeta):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, tokenizer, shared_config: SharedConfig):
        self.batch_size: int = dl_config.batch_size
        self.num_workers: int = dl_config.num_workers
        self.pin_memory: bool = dl_config.pin_memory
        self.drop_last: bool = dl_config.drop_last
        self.shuffle: bool = dl_config.shuffle
        self.tokenizer: Dict[str] = tokenizer
        self.vocab_transform: Dict[str] = {}
        self.text_transform: Dict[str] = {}
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

    def build_vocab(self):
        vocab_transform = {}
        for ln in [self.src_language, self.tgt_language]:
            vocab_transform[ln] = Vocabulary(ln)
            train_iter = self.train_dataset
            for tokens in self.yield_tokens(train_iter, ln):
                vocab_transform[ln].add_words(tokens)
        return vocab_transform

    def yield_tokens(self, data_iter: list, language: str):
        language_index = {self.src_language: 0, self.tgt_language: 1}
        for data_sample in data_iter:
            yield self.tokenizer[language](data_sample[language_index[language]])

    def collate_fn(self, batch: List[str]) -> Tuple[List[str], List[str]]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.src_language](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.tgt_language](tgt_sample.rstrip("\n")))

        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=self.special_symbols.index('<pad>'))
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.special_symbols.index('<pad>'))

        return src_batch, tgt_batch

    def data_iter(self, iterator):
        for data in iterator:
            yield data


class IWSLT2017DataLoader(BaseDataLoader):
    def __init__(self, dl_config: DataLoaderConfig, tkn_config: TokenizerConfig, tokenizer: Any, shared_config: SharedConfig):
        super().__init__(dl_config, tokenizer, tkn_config, shared_config)
        
        self.dataset = load_dataset("iwslt2017", f'iwslt2017-{self.src_language}-{self.tgt_language}', cache_dir='./.data/iwslt2017')
            
        self.build_datasets()
        self.logger.info('Datasets have benn loaded.')
            
        self.vocab_transform = super().build_vocab()
        self.logger.info('Vcabulary have benn built.')
            
        self.text_transform = create_text_transform(self.src_language, self.tgt_language, 
                                                        self.tokenizer, self.vocab_transform)
        self.logger.info('Text Transform have been instantiated.')

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
            
        self.vocab_transform = super().build_vocab()
        self.logger.info('Vcabulary have benn built.')
            
        self.text_transform = create_text_transform(self.src_language, self.tgt_language, 
                                                        self.tokenizer, self.vocab_transform)
        self.logger.info('Text Transform have been instantiated.')

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
        

def create_text_transform(src_lang: str, tgt_lang: str, tokenizer: Any, vocab_transform: Dict[str, str]) -> Any:
    text_transform = {}
    for ln in [src_lang, tgt_lang]:
        text_transform[ln] = sequential_transforms(tokenizer[ln],
                                                   vocab_transform[ln].to_index,
                                                   tensor_transform)
    return text_transform

def sequential_transforms(*transforms: Any):
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
    
def load_vocab(run_id: str):
        vocab_file_path = f'./results/{run_id}/vocab.pth'
        
        return torch.load(vocab_file_path)
        