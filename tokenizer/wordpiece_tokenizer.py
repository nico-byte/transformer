from typing import List
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
import torch
from torchtext.datasets import Multi30k
from datasets import load_dataset
import os

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()


def build_tokenizer(name: str, run_id: str, dataset: List[str], vocab_size: int):

    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )
    
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
    )

    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))

    if not os.path.exists(f"./models/{run_id}/tokenizer"):
        os.makedirs(f"./models/{run_id}/tokenizer")
    
    tokenizer_path = f"./models/{run_id}/tokenizer/{name}.json"
    tokenizer.save(tokenizer_path)
    
    return tokenizer_path

def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]
