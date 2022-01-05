from pathlib import Path
import os

DATASET = "yelp"
DATA_PATH = Path("../data") / DATASET

neg_train = (DATA_PATH / "sentiment.train.0").read_text().split("\n")
pos_train = (DATA_PATH / "sentiment.train.1").read_text().split("\n")
neg_valid = (DATA_PATH / "sentiment.dev.0").read_text().split("\n")
pos_valid = (DATA_PATH / "sentiment.dev.1").read_text().split("\n")

lines_train = [f"__label__pos {l}"for l in pos_train] + [f"__label__neg {l}"for l in neg_train]
lines_valid = [f"__label__pos {l}"for l in pos_valid] + [f"__label__neg {l}"for l in neg_valid]

with open(f"{DATASET}.train.txt", "w") as f:
    f.write("\n".join(lines_train))

with open(f"{DATASET}.valid.txt", "w") as f:
    f.write("\n".join(lines_valid))

os.system(f"cat {DATASET}.train.txt | sed -e \"s/\([.\!?,’/()]\)/ \\1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > {DATASET}.train.processed.txt")
os.system(f"cat {DATASET}.valid.txt | sed -e \"s/\([.\!?,’/()]\)/ \\1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > {DATASET}.valid.processed.txt")