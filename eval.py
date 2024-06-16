import warnings
import argparse
from evaluate import load as load_metric
import json

from src.data import IWSLT2017DataLoader
from utils.config import SharedConfig, DataLoaderConfig
from src.processor import Processor

warnings.filterwarnings("ignore", category=UserWarning)


def parsing_args():
    parser = argparse.ArgumentParser(description="Parsing some important arguments.")
    parser.add_argument("path_to_checkpoint", type=str)
    parser.add_argument("path_to_tokenizer", type=str)
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
    )

    return parser.parse_args()


def main(args):
    path_to_checkpoint = args.path_to_checkpoint
    path_to_tokenizer = args.path_to_tokenizer
    device = args.torch_device

    shared_conf = SharedConfig()
    dl_conf = DataLoaderConfig()

    dataloader = IWSLT2017DataLoader(dl_conf, shared_conf)

    val_dataset = dataloader.val_dataset

    translator = Processor.from_checkpoint(
        model_checkpoint=path_to_checkpoint, tokenizer=path_to_tokenizer, device=device
    )

    bleu = load_metric("bleu")
    sacre_bleu = load_metric("sacrebleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")

    outputs = []
    sources = [x[0] for x in val_dataset]
    targets = [x[1] for x in val_dataset]

    for idx, src in enumerate(sources):
        output = translator.translate(src)

        outputs.append(output)

        print(f"{idx+1}/{len(sources)}", end="\r")

    bleu_score = bleu.compute(predictions=outputs, references=targets)

    sacre_bleu_score = sacre_bleu.compute(predictions=outputs, references=targets)

    rouge_score = rouge.compute(predictions=outputs, references=targets)

    meteor_score = meteor.compute(predictions=outputs, references=targets)

    metrics = {
        "bleu": bleu_score,
        "sacre_bleu": sacre_bleu_score,
        "rouge": rouge_score,
        "meteor": meteor_score,
    }

    # Convert and write JSON object to file
    with open(
        f"./{shared_conf.src_language}test-{shared_conf.tgt_language}-metrics.json", "x"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)

    print(
        f"\n\nEvaluation: bleu_score - {bleu_score}\nEvaluation: rouge_score - {rouge_score}\nEvaluation: sacre_bleu_score - {sacre_bleu_score}\nEvaluation: meteor_score - {meteor_score}"
    )

    TEST_SEQUENCE = (
        "The quick brown fox jumped over the lazy dog and then ran away quickly."
    )
    output = translator.translate(TEST_SEQUENCE)

    print(f"Input: {TEST_SEQUENCE}, Output: {output}")


if __name__ == "__main__":
    args = parsing_args()
    main(args)
