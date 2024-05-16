from typing import List, Dict, Any
from pydantic import BaseModel as PydanticBaseModel
import torch
from torchtext.data.utils import get_tokenizer


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class SharedStore(BaseModel):
    token_transform: Dict[str, Any]
    text_transform: Dict[str, Any] = {}
    vocab_transform: Dict[str, Any] = {}
    dataloaders: List[Any] = []
    special_symbols: List[str] = ['<unk>', '<bos>', '<eos>', '<pad>']


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
    

class TransformerConfig(BaseModel):
    num_encoder_layers: int
    num_decoder_layers: int
    emb_size: int
    nhead: int
    src_vocab_size: int
    tgt_vocab_size: int
    dim_feedforward: int = 512
    dropout: float = 0.1
    shared_store: SharedStore


class TrainerConfig(BaseModel):
    learning_rate: float
    num_epochs: int
    batch_size: int
    tgt_batch_size: int = None
    num_cycles: int = None
    stepsize: int = None
    device: torch.device = None
