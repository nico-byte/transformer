from typing import List
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
import torch
import os
import random

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()


def build_tokenizer(name: str, run_id: str, src_dataset: List[str], tgt_dataset: List[str], vocab_size: int):
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
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
    )

    # Combine the source and target datasets
    # combined_dataset = src_dataset + tgt_dataset

    # Shuffle the combined dataset to ensure a balanced representation
    # random.shuffle(combined_dataset)

    # Train the tokenizer on the combined dataset
    # tokenizer.train_from_iterator(batch_iterator(tgt_dataset), trainer=trainer, length=len(combined_dataset))
    tokenizer.train_from_iterator(batch_iterator(src_dataset), trainer=trainer, length=len(combined_dataset))

    if not os.path.exists(f'./models/{run_id}/'):
        os.makedirs(f'./models/{run_id}/')

    tokenizer.save(f"./models/{run_id}/tokenizer-{name}.json")
    
    return tokenizer

def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]
