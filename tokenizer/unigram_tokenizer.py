from typing import List
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
import os
import random


def build_tokenizer(name: str, run_id: str, src_dataset: List[str], tgt_dataset: List[str], vocab_size: int):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
    )

    # Combine the source and target datasets
    combined_dataset = src_dataset + tgt_dataset

    # Shuffle the combined dataset to ensure a balanced representation
    random.shuffle(combined_dataset)

    # Train the tokenizer on the combined dataset
    tokenizer.train_from_iterator(batch_iterator(combined_dataset), trainer=trainer, length=len(combined_dataset))

    if not os.path.exists(f'./models/{run_id}/'):
        os.makedirs(f'./models/{run_id}/')

    tokenizer.save(f"./models/{run_id}/tokenizer-{name}.json")
    
    return tokenizer
    
def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]
