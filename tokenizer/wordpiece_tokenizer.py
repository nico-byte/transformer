from typing import List
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tokenizers.processors import TemplateProcessing
import torch
import os
import random

# in case error occurs that it cant be imported by torch
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = (
    torch.utils._import_utils.dill_available()
)


def build_tokenizer(
    run_id: str, src_dataset: List[str], tgt_dataset: List[str], vocab_size: int
) -> Tokenizer:
    """
    Build and train a tokenizer on the provided source and target datasets.

    Args:
        name (str): The name to save the tokenizer under.
        run_id (str): The run identifier for saving the tokenizer.
        src_dataset (List[str]): The source dataset for tokenization.
        tgt_dataset (List[str]): The target dataset for tokenization.
        vocab_size (int): The vocabulary size for the tokenizer.

    Returns:
        tokenizers.Tokenizer: The trained tokenizer.

    The function combines the source and target datasets, shuffles them, and trains a wordpiece tokenizer.
    The tokenizer is configured with normalization, pre-tokenization, and post-processing steps, and is then saved
    to the specified directory under the given run ID.
    """
    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
    )

    # Combine the source and target datasets
    combined_dataset = src_dataset + tgt_dataset

    # Shuffle the combined dataset to ensure a balanced representation
    random.shuffle(combined_dataset)

    # Train the tokenizer on the combined dataset
    tokenizer.train_from_iterator(
        batch_iterator(combined_dataset), trainer=trainer, length=len(combined_dataset)
    )

    if not os.path.exists(f"./models/{run_id}/"):
        os.makedirs(f"./models/{run_id}/")

    tokenizer.save(f"./models/{run_id}/tokenizer.json")

    return tokenizer


def batch_iterator(dataset, batch_size=1000):
    """
    Batch iterator to yield batches of data from the dataset.

    Args:
        dataset (List[str]): The dataset to iterate over.
        batch_size (int, optional): The size of each batch. Defaults to 1000.

    Yields:
        List[str]: A batch of data from the dataset.

    The function splits the dataset into batches of the specified size and yields each batch.
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]
