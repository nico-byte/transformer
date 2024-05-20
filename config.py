from typing import List, Dict, Any
from pydantic import BaseModel as PydanticBaseModel
import torch
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torchtext.data.utils import get_tokenizer


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class SharedConfig(BaseModel):
    special_symbols: List[str] = ['<pad>', '<eos>', '<bos>', '<unk>']


class TokenizerConfig(BaseModel):
    src_language: str = 'de'
    tgt_language: str = 'en'
    src_tokenizer: Any = get_tokenizer('spacy', language='de_dep_news_trf')
    tgt_tokenizer: Any = get_tokenizer('spacy', language='en_core_web_trf')


class DataLoaderConfig(PydanticBaseModel):
    dataset: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    shuffle: bool
    

class TransformerConfig(PydanticBaseModel):
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    emb_size: int = 512
    nhead: int = 8
    src_vocab_size: int
    tgt_vocab_size: int
    dim_feedforward: int = 512
    dropout: float = 0.1

class TrainerConfig(PydanticBaseModel):
    learning_rate: float
    num_epochs: int
    batch_size: int
    tgt_batch_size: int = None
    num_cycles: int = None
