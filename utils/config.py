from typing import List
from pydantic import BaseModel as PydanticBaseModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class SharedConfig(BaseModel):
    special_symbols: List[str] = ['<unk>', '<bos>', '<eos>', '<pad>']
    run_id: str = None


class TokenizerConfig(BaseModel):
    src_language: str = 'en'
    tgt_language: str = 'de'


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
    warmup_steps: int = None
