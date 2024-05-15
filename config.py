from typing import List, Dict, Any
from pydantic import BaseModel
from torchtext.data.utils import get_tokenizer


class SharedStore(BaseModel):
    special_symbols: List[str] = ['<unk>', '<bos>', '<eos>', '<pad>']
    text_transform: Any = None
    token_transform: Dict[str, Any]
    vocab_transform: Any = None
    dataloaders: List[Any] = []


class TokenizerConfig(BaseModel):
    src_language: str = 'de'
    tgt_language: str = 'en'
    src_tokenizer: Any = get_tokenizer('spacy', language='de_core_news_sm')
    tgt_tokenizer: Any = get_tokenizer('spacy', language='en_core_web_sm')


class DataLoaderConfig(BaseModel):
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
    shared_store: Any


class TrainerConfig(BaseModel):
    learning_rate: float
    num_epochs: int
    batch_size: int
    tgt_batch_size: int = None
    num_cycles: int = None
    stepsize: int = None
    device: Any = None
