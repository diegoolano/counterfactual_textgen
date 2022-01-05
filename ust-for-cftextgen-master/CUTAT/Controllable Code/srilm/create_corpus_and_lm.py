from pathlib import Path
import os

DATASET = "yelp"
DATA_PATH = Path("../data") / DATASET / "processed_files"

id_to_word = (DATA_PATH / "word_to_id.txt").read_text().split("\n")
id_to_word = [x.split("\t")[0] for x in id_to_word]
neg_train = [" ".join([id_to_word[int(i)] for i in line.strip().split()]) for line in (DATA_PATH / "sentiment.train.0").read_text().split("\n")]
neg_test = [" ".join([id_to_word[int(i)] for i in line.strip().split()]) for line in (DATA_PATH / "sentiment.test.0").read_text().split("\n")]
neg_valid = [" ".join([id_to_word[int(i)] for i in line.strip().split()]) for line in (DATA_PATH / "sentiment.dev.0").read_text().split("\n")]
pos_train = [" ".join([id_to_word[int(i)] for i in line.strip().split()]) for line in (DATA_PATH / "sentiment.train.1").read_text().split("\n")]
pos_test = [" ".join([id_to_word[int(i)] for i in line.strip().split()]) for line in (DATA_PATH / "sentiment.test.1").read_text().split("\n")]
pos_valid = [" ".join([id_to_word[int(i)] for i in line.strip().split()]) for line in (DATA_PATH / "sentiment.dev.1").read_text().split("\n")]

corpus = neg_train + neg_test + neg_valid + pos_train + pos_test + pos_valid

with open(f"{DATASET}.corpus.txt", "w") as f:
    f.write("\n".join(corpus))

os.system(f"/usr/share/srilm/bin/i686-m64/ngram-count  -text  "
          f"./{DATASET}.corpus.txt -order 2 -addsmooth 0 -lm  "
          f"./{DATASET}.corpus.lm")