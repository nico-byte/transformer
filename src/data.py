from typing import List, Tuple
from utils.logger import get_logger
import abc
import random
from src.pretrained_inference import mt_batch_inference
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from tokenizers import Tokenizer

from torchtext.datasets import Multi30k
from datasets import load_dataset
from utils.config import DataLoaderConfig, SharedConfig
from tokenizer.wordpiece_tokenizer import build_tokenizer as build_wordpiece_tokenizer
from tokenizer.unigram_tokenizer import build_tokenizer as build_unigram_tokenizer

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()


class BaseDataLoader(metaclass=abc.ABCMeta):
    """
    Abstract base class for data loaders in the application. Provides common functionality for building datasets and dataloaders.

    Attributes:
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of worker processes to use for data loading.
        pin_memory (bool): Whether to use pinned memory for faster data transfer.
        drop_last (bool): Whether to drop the last incomplete batch.
        shuffle (bool): Whether to shuffle the data.
        tokenizer (Tokenizer): The tokenizer to use for encoding the input data.
        src_language (str): The source language.
        tgt_language (str): The target language.
        special_symbols (List[str]): A list of special symbols to use in the tokenizer.
        train_dataset (List[Tuple[str, str]]): The training dataset.
        val_dataset (List[Tuple[str, str]]): The validation dataset.
        test_dataset (List[Tuple[str, str]]): The test dataset.
        train_dataloader (DataLoader): The training dataloader.
        test_dataloader (DataLoader): The test dataloader.
        val_dataloader (DataLoader): The validation dataloader.
        logger (Logger): The logger for the data loader.

    Methods:
        build_datasets(): Abstract method for building the datasets.
        build_dataloaders(): Builds the dataloaders for the training, validation, and test datasets.
        collate_fn(batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]: Collate function to process a batch of data.
        backtranslate_dataset(): Performs back-translation on the training dataset.
        clean_dataset(dataset: List[Tuple[str, str]]) -> List[Tuple[str, str]]: Cleans the dataset by removing duplicates and empty sequences.
        train_tokenizer(run_id: str, vocab_size: int, tokenizer: str="wordpiece"): Trains the tokenizer on the dataset.
        batch_iterator(dataset, batch_size=1000): Yields batches of the dataset.
    """
    def __init__(
        self, dl_config: DataLoaderConfig, shared_config: SharedConfig
        ):
        self.batch_size: int = dl_config.batch_size
        self.num_workers: int = dl_config.num_workers
        self.pin_memory: bool = dl_config.pin_memory
        self.drop_last: bool = dl_config.drop_last
        self.shuffle: bool = dl_config.shuffle
        self.tokenizer = None
        self.src_language: str = shared_config.src_language
        self.tgt_language: str = shared_config.tgt_language
        self.special_symbols: List[str] = shared_config.special_symbols
        
        self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
        self.train_dataloader, self.test_dataloader, self.val_dataloader = None, None, None
        
        self.logger = get_logger('DataLoader')

    @abc.abstractmethod
    def build_datasets(self):
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        pass

    def build_dataloaders(self):
        """
        Build DataLoaders for the training, validation, and test datasets.
        """
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

    def collate_fn(
        self, batch: List[Tuple[str, str]]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to process a batch of data.

        Args:
            batch (List[Tuple[str, str]]): Batch of source and target sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded source and target batches as tensors.
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            encoded_src_sample = self.tokenizer.encode(src_sample)
            tensor_src_sample = torch.tensor(encoded_src_sample.ids)
            src_batch.append(tensor_src_sample)
            
            encoded_tgt_sample = self.tokenizer.encode(tgt_sample)
            tensor_tgt_sample = torch.tensor(encoded_tgt_sample.ids)
            tgt_batch.append(tensor_tgt_sample)


        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=3)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=3)

        return src_batch, tgt_batch
    
    def backtranslate_dataset(self):
        """
        Backtranslate the dataset to augment the training data.
        
        This method takes the target sequences from the training dataset, backtranslates them using a machine translation model, and then adds the backtranslated sequences as new training examples.
        
        The resulting augmented dataset is then cleaned to remove any duplicate or empty sequences before being assigned back to the `self.train_dataset` attribute.
        """
        tgt_dataset = [x[1] for x in self.train_dataset]
        
        backtrans_dataset = mt_batch_inference(tgt_dataset, "cuda", 512, self.logger)
        
        backtrans_dataset_pairs = [[x, y] for x, y in zip(backtrans_dataset, tgt_dataset)]
        
        new_dataset = self.train_dataset + backtrans_dataset_pairs
        self.train_dataset = self.clean_dataset(new_dataset)
        
    def clean_dataset(
        self, dataset: List[Tuple[str, str]]
        ):
        """
        Clean the dataset by removing duplicates and empty sequences.

        Args:
            dataset (List[Tuple[str, str]]): Dataset to be cleaned.

        Returns:
            List[Tuple[str, str]]: Cleaned dataset.
        """

        src_dataset = [x[0] for x in dataset]
        tgt_dataset = [x[1] for x in dataset]
        
        unique_pairs = {}
        for x, y in zip(src_dataset, tgt_dataset):
            if (x and y) and (x not in unique_pairs):
                unique_pairs[x] = y
    
        clean_dataset = [[x, y] for x, y in unique_pairs.items()]
        self.logger.info(f'Cleaned dataset: {len(clean_dataset)}')
    
        return clean_dataset
    
    def train_tokenizer(
        self, run_id: str, vocab_size: int, tokenizer: str="wordpiece"
        ):
        """        
        This method loads the IWSLT2017 dataset, extracts the source and target language sequences, and then builds a tokenizer (either WordPiece or Unigram) using the training sequences. The trained tokenizer is then assigned to the `self.tokenizer` attribute.
        
        Args:
            run_id (str): A unique identifier for the current run.
            vocab_size (int): The desired vocabulary size for the tokenizer.
            tokenizer (str, optional): The type of tokenizer to use, either "wordpiece" or "unigram". Defaults to "wordpiece".
        """
        dataset = load_dataset("iwslt2017", f'iwslt2017-{self.src_language}-{self.tgt_language}', cache_dir='./.data/iwslt2017')
        dataset = [(d[self.src_language], d[self.tgt_language]) for d in dataset["train"]['translation']]

        src_dataset = [x[0] for x in dataset]
        tgt_dataset = [x[1] for x in dataset]
        
        if tokenizer == "wordpiece":
            self.tokenizer = build_wordpiece_tokenizer(run_id, src_dataset, tgt_dataset, vocab_size)
        else:
            self.tokenizer = build_unigram_tokenizer(run_id, src_dataset, tgt_dataset, vocab_size)

    @staticmethod
    def batch_iterator(
        dataset: Tuple[List[str, str]], batch_size: int = 1000
        ):
        """
        Generates batches of the given dataset.
    
        Args:
            dataset (Tuple[List[str, str]]): The dataset to be batched.
            batch_size (int, optional): The size of each batch. Defaults to 1000.
    
        Yields:
            Tuple[List[str, str]]: A batch of the dataset.
        """
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]


class IWSLT2017DataLoader(BaseDataLoader):
    """
    Build the datasets for training, validation, and testing from the IWSLT2017 dataset.

    This method loads the IWSLT2017 dataset, extracts the source and target language sequences, and assigns them to the `self.train_dataset`, `self.test_dataset`, and `self.val_dataset` attributes.

    The training dataset is loaded from the "train" split, the test dataset is loaded from the "test" split, and the validation dataset is loaded from the "validation" split. The dataset entries are extracted as tuples of (source, target) language sequences.

    Some debug logging is also performed to print the first entry and length of each dataset.
    """
    def __init__(
        self, dl_config: DataLoaderConfig, shared_config: SharedConfig
        ):
        """
        Initializes the IWSLT2017DataLoader class, which is responsible for loading and preparing the IWSLT2017 dataset for use in a machine learning model.
        
        The constructor takes two arguments:
            - `dl_config`: a `DataLoaderConfig` object that contains configuration settings for the data loader
            - `shared_config`: a `SharedConfig` object that contains shared configuration settings across the application
        
        The constructor performs the following steps:
        1. Calls the constructor of the parent class (`BaseDataLoader`) with the provided `dl_config` and `shared_config` arguments.
        2. Loads the IWSLT2017 dataset using the `load_dataset` function from the `datasets` library, with the dataset name `'iwslt2017'` and the dataset configuration `f'iwslt2017-{self.src_language}-{self.tgt_language}'`. The dataset is cached in the `./.data/iwslt2017` directory.
        3. Calls the `build_datasets` method to initialize the `self.train_dataset`, `self.test_dataset`, and `self.val_dataset` attributes with the loaded dataset.
        4. Logs a message indicating that the datasets have been loaded.
        """
        super().__init__(dl_config, shared_config)
        
        self.dataset = load_dataset("iwslt2017", f'iwslt2017-{self.src_language}-{self.tgt_language}', cache_dir='./.data/iwslt2017')
            
        self.build_datasets()
        self.logger.info('Datasets have been loaded.')

    @classmethod
    def build_with_tokenizer(
        cls, dl_config: DataLoaderConfig, shared_config: SharedConfig, tokenizer: str
        ):
        """
        Builds a new instance of the `Multi30kDataLoader` class with a pre-trained tokenizer.

        Args:
            dl_config (DataLoaderConfig): The data loader configuration.
            shared_config (SharedConfig): The shared configuration.
            tokenizer (str): The path to the pre-trained tokenizer file.

        Returns:
            Multi30kDataLoader: A new instance of the `Multi30kDataLoader` class with the specified tokenizer.
        """
        dataloader = cls(dl_config, shared_config)

        dataloader.tokenizer = Tokenizer.from_file(tokenizer)

        super().build_dataloaders(dataloader)
        dataloader.logger.info("Dataloaders have been built.")

        return dataloader

    @classmethod
    def new_instance(
        cls,
        dl_config: DataLoaderConfig,
        shared_config: SharedConfig,
        tokenizer: str = "wordpiece",
        ):
        """
        Builds a new instance of the `Multi30kDataLoader` class with a pre-trained tokenizer.
    
        Args:
            dl_config (DataLoaderConfig): The data loader configuration.
            shared_config (SharedConfig): The shared configuration.
            tokenizer (str, optional): The path to the pre-trained tokenizer file. Defaults to "wordpiece".
    
        Returns:
            Multi30kDataLoader: A new instance of the `Multi30kDataLoader` class with the specified tokenizer.
        """
        dataloader = cls(dl_config, shared_config)

        super().train_tokenizer(dataloader, shared_config.run_id, 3280, tokenizer)

        super().build_dataloaders(dataloader)
        dataloader.logger.info("Dataloaders have been built.")

        return dataloader
        
    def build_datasets(self):
        """
        Builds the training, validation, and test datasets for the Multi30k dataset.
        
        The datasets are constructed by extracting the source and target language pairs from the
        "translation" field of the dataset. The training dataset is constructed from the "train"
        split, the test dataset is constructed from the "test" split, and the validation dataset
        is constructed from a random 5% sample of the training dataset.
        
        The first entry and length of each dataset are logged for debugging purposes.
        """
        self.train_dataset: List[str, str] = [(d[self.src_language], d[self.tgt_language]) for d in self.dataset["train"]['translation']]
        self.test_dataset: List[str, str] = [(d[self.src_language], d[self.tgt_language]) for d in self.dataset["test"]['translation']]
        self.val_dataset: List[str, str] = [(d[self.src_language], d[self.tgt_language]) for d in self.dataset["validation"]['translation']]
        
        self.logger.debug("First Entry train dataset: %s", list(self.train_dataset[0]))
        self.logger.debug("Length train dataset: %f", len(self.train_dataset))
        self.logger.debug("First Entry test dataset: %s", list(self.test_dataset[0]))
        self.logger.debug("Length test dataset: %f", len(self.test_dataset))
        self.logger.debug("First Entry val dataset: %s", list(self.val_dataset[0]))
        self.logger.debug("Length val dataset: %f", len(self.val_dataset))


class Multi30kDataLoader(BaseDataLoader):
    """
    Builds the training, validation, and test datasets for the Multi30k dataset.
    
    The datasets are constructed by extracting the source and target language pairs from the
    "translation" field of the dataset. The training dataset is constructed from the "train"
    split, the test dataset is constructed from the "valid" split, and the validation dataset
    is constructed from a random 5% sample of the training dataset.
    
    The first entry and length of each dataset are logged for debugging purposes.
    """
    def __init__(
        self, dl_config: DataLoaderConfig, shared_config: SharedConfig
        ):
        """
        Initializes the Multi30kDataLoader class, which is responsible for loading and preparing the IWSLT2017 dataset for use in a machine learning model.
        
        The constructor takes two arguments:
            - `dl_config`: a `DataLoaderConfig` object that contains configuration settings for the data loader
            - `shared_config`: a `SharedConfig` object that contains shared configuration settings across the application
        
        The constructor performs the following steps:
        1. Calls the constructor of the parent class (`BaseDataLoader`) with the provided `dl_config` and `shared_config` arguments.
        2. Calls the `build_datasets` method to initialize the Multi30k dataset, `self.train_dataset`, `self.test_dataset`, and `self.val_dataset` attributes with the loaded dataset.
        3. Logs a message indicating that the datasets have been loaded.
        """
        super().__init__(dl_config, shared_config)

        self.build_datasets()
        self.logger.info('Datasets have benn loaded.')

    @classmethod
    def build_with_tokenizer(
        cls, dl_config: DataLoaderConfig, shared_config: SharedConfig, tokenizer: str
        ):
        """
        Builds a new instance of the `Multi30kDataLoader` class with a specified tokenizer.
    
        This class method is responsible for creating a new instance of the `Multi30kDataLoader` class, initializing the tokenizer, and building the necessary dataloaders for the Multi30k dataset.
    
        Args:
            dl_config (DataLoaderConfig): A configuration object containing settings for the data loader.
            shared_config (SharedConfig): A configuration object containing shared settings across the application.
            tokenizer (str): The name of the tokenizer to use, defaults to "wordpiece".
    
        Returns:
            Multi30kDataLoader: A new instance of the `Multi30kDataLoader` class with the specified configurations and tokenizer.
        """
        dataloader = cls(dl_config, shared_config)

        dataloader.tokenizer = Tokenizer.from_file(tokenizer)

        super().build_dataloaders(dataloader)
        dataloader.logger.info("Dataloaders have been built.")

        return dataloader

    @classmethod
    def new_instance(
        cls,
        dl_config: DataLoaderConfig,
        shared_config: SharedConfig,
        tokenizer: str = "wordpiece",
        ):
        """
        Builds a new instance of the `Multi30kDataLoader` class with a specified tokenizer.
    
        This class method is responsible for creating a new instance of the `Multi30kDataLoader` class, initializing the tokenizer, and building the necessary dataloaders for the Multi30k dataset.
    
        Args:
            dl_config (DataLoaderConfig): A configuration object containing settings for the data loader.
            shared_config (SharedConfig): A configuration object containing shared settings across the application.
            tokenizer (str): The name of the tokenizer to use, defaults to "wordpiece".
    
        Returns:
            Multi30kDataLoader: A new instance of the `Multi30kDataLoader` class with the specified configurations and tokenizer.
        """
        dataloader = cls(dl_config, shared_config)

        super().train_tokenizer(dataloader, shared_config.run_id, 1640, tokenizer)

        super().backtranslate_dataset(dataloader)

        super().build_dataloaders(dataloader)
        dataloader.logger.info("Dataloaders have been built.")

        return dataloader
        
    def build_datasets(self):
        """
        Builds the training, validation, and test datasets for the Multi30k dataset.
        
        This method is responsible for loading the Multi30k dataset, splitting it into training, validation, and test sets, and storing them as attributes of the `Multi30kDataLoader` instance.
        
        The training dataset is loaded from the 'train' split of the Multi30k dataset, and 5% of the training samples are randomly selected to create the validation dataset. The remaining samples are kept in the training dataset.
        
        The test dataset is loaded from the 'valid' split of the Multi30k dataset.
        
        The method also logs some debug information about the loaded datasets, including the first entry and the length of each dataset.
        """
        self.train_dataset: List[str, str] = list(Multi30k(root='./.data/multi30k', split='train',
                                      language_pair=(self.src_language, self.tgt_language)))
        
        self.test_dataset: List[str, str] = list(Multi30k(root='./.data/multi30k',  split='valid',
                                    language_pair=(self.src_language, self.tgt_language)))
        
        total_entries = len(self.train_dataset)
        num_test_entries = int(total_entries * 0.05)
        val_indices = random.sample(range(total_entries), num_test_entries)
        
        self.val_dataset: List[str, str] = [self.train_dataset[i] for i in val_indices]
        self.train_dataset = [entry for i, entry in enumerate(self.train_dataset) if i not in val_indices]
        
        self.logger.debug("First Entry train dataset: %s", list(self.train_dataset[0]))
        self.logger.debug("Length train dataset: %f", len(self.train_dataset))
        self.logger.debug("First Entry test dataset: %s", list(self.test_dataset[0]))
        self.logger.debug("Length test dataset: %f", len(self.test_dataset))
        self.logger.debug("First Entry val dataset: %s", list(self.val_dataset[0]))
        self.logger.debug("Length val dataset: %f", len(self.val_dataset))
        
        