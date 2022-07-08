from glob import glob
import json
import os
import argparse
import random


def parse_keyphrases(text, keywords):
    keyphrases = []
    for keyphrase in keywords:
        keyphrase_index = text.find(keyphrase)
        if keyphrase_index == -1:
            keyphrase_index = text.lower().find(keyphrase)
            # Can't find keyphrase in text
            if keyphrase_index == -1:
                continue
            keyphrase = text[keyphrase_index : keyphrase_index + len(keyphrase)]
        # Decide whether a space is required before for the tokenizer to have a consistent behavior
        if keyphrase_index > 0:
            if text[keyphrase_index - 1] == " ":
                keyphrase = " " + keyphrase
        keyphrases.append(keyphrase)

    return keyphrases


def main(args):
    random.seed(42)
    keyphrase_universe = []
    max_keyphrase_pairs = 0
    corpus_dirs = [args.train_data_dir]
    for corpus_dir in corpus_dirs:
        data_files = glob(corpus_dir + "/*")

        for fname in data_files:
            with open(fname) as f:
                for line in f:
                    data = json.loads(line)
                    if "keywords" in data:
                        title = data["title"]
                        abstract = data["abstract"]
                        text = title + ". " + abstract
                        keywords = data["keywords"].split(" , ")
                        keyphrases = parse_keyphrases(text, keywords)
                        if len(keyphrases) > max_keyphrase_pairs:
                            max_keyphrase_pairs = len(keyphrases)
                        keyphrase_universe += keyphrases

    print("Max Keyphrase Pairs: ", max_keyphrase_pairs)
    keyphrase_universe = list(set(keyphrase_universe))
    random.shuffle(keyphrase_universe)
    with open(os.path.join(args.output_dir, "keyphrase_universe.txt"), "w+") as outf:
        for keyphrase in keyphrase_universe:
            outf.write(keyphrase)
            outf.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-dir",
        type=str,
        help="Train files from which the keyphrase universe should be computed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output file containing all keyphrases from the corpus",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
