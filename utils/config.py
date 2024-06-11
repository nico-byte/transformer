from typing import List
from pydantic import BaseModel as PydanticBaseModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseModel(PydanticBaseModel):
    """
    A base model extending Pydantic's BaseModel to allow arbitrary types.

    Config:
        arbitrary_types_allowed (bool): Allows arbitrary types to be used in the model.
    """

    class Config:
        arbitrary_types_allowed = True


class SharedConfig(BaseModel):
    """
    A configuration class for shared settings.

    Attributes:
        special_symbols (List[str]): Special symbols used in tokenization, default is ['<unk>', '<bos>', '<eos>', '<pad>'].
        run_id (str): Identifier for the run, default is None.
    """

    special_symbols: List[str] = ['<unk>', '<bos>', '<eos>', '<pad>']
    run_id: str = None


class TokenizerConfig(BaseModel):
    """
    A configuration class for the tokenizer settings.

    Attributes:
        src_language (str): Source language code, default is 'en'.
        tgt_language (str): Target language code, default is 'de'.
    """

    src_language: str = 'en'
    tgt_language: str = 'de'


class DataLoaderConfig(PydanticBaseModel):
    """
    A configuration class for data loader settings.

    Attributes:
        dataset (str): Name of the dataset.
        batch_size (int): Size of the batches.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        drop_last (bool): If True, drops the last incomplete batch.
        shuffle (bool): If True, shuffles the data every epoch.
    """

    dataset: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    shuffle: bool
    

class TransformerConfig(PydanticBaseModel):
    """
    A configuration class for the transformer model settings.

    Attributes:
        num_encoder_layers (int): Number of encoder layers, default is 4.
        num_decoder_layers (int): Number of decoder layers, default is 4.
        emb_size (int): Size of the embeddings, default is 512.
        nhead (int): Number of attention heads, default is 8.
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        dim_feedforward (int): Dimension of the feedforward network, default is 512.
        dropout (float): Dropout rate, default is 0.1.
    """

    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    emb_size: int = 512
    nhead: int = 8
    src_vocab_size: int
    tgt_vocab_size: int
    dim_feedforward: int = 512
    dropout: float = 0.1

class TrainerConfig(PydanticBaseModel):
    """
    A configuration class for training settings.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Size of the batches.
        tgt_batch_size (int): Size of the target batches, default is None.
        warmup_steps (int): Number of warmup steps for learning rate scheduling, default is None.
    """

    learning_rate: float
    num_epochs: int
    batch_size: int
    tgt_batch_size: int = None
    warmup_steps: int = None
