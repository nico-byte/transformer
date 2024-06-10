from typing import List, Optional
from pydantic import BaseModel as PydanticBaseModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class SharedConfig(BaseModel):
    special_symbols: List[str] = ['<unk>', '<bos>', "<eos>", "<pad>"]
    run_id: Optional[str] = None


class TokenizerConfig(BaseModel):
    src_language: str = 'en'
    tgt_language: str = 'de'


class DataLoaderConfig(PydanticBaseModel):
    dataset: str = 'iwslt2017'
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    

class TransformerConfig(PydanticBaseModel):
    num_encoder_layers: int
    num_decoder_layers: int
    emb_size: int
    nhead: int
    src_vocab_size: int
    tgt_vocab_size: int
    dim_feedforward: int 
    dropout: float

class TrainerConfig(PydanticBaseModel):
    learning_rate: float
    num_epochs: int
    batch_size: int
    tgt_batch_size: int
    warmup_steps: int
