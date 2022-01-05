import argparse
import os
import csv
import pandas as pd
from tqdm import tqdm, trange

"""
step1_bert_classification_training_data_preparation.py

# LIPTONS data looks like this
diego@microdeep:~/spr20_cf_gen/TDRG/data/lipton-data/sentiment/orig$ head train.tsv
Sentiment       Text
Negative        Long, boring, blasphemous. Never have I been so glad to see ending credits roll.
Negative        Not good! Rent or buy the original! Watch this only if someone has 

# Whereas what it created BEFORE looked like this
diego@microdeep:~/spr20_cf_gen/TDRG/data/yelp/bert_classifier_training$ head train.csv 
i was sadly mistaken .  0
so on to the hoagies , the italian is general run of the mill . 0
minimal meat and a ton of shredded lettuce .    0

# so we want to make liptons look like the yelp output so we can feed this into run_classifier.py

"""
def create_classification_file(input_df, output_file_path):
    """
    Create a csv file combining training data for BERT classification training.
    input_df : input dataframe
    output_file_path : csv file path
    """
    with open(output_file_path, "w") as out_fp:
        writer = csv.writer(out_fp, delimiter="\t")
        for i in tqdm(range(input_df.shape[0])):
            line = input_df.Text.values[i]
            cur_label = 0 if input_df.Sentiment.values[i] == "Negative" else 1
            writer.writerow([line.strip(),cur_label])

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument( "--data_dir", default="../data/", type=str, required=False, help="")
    parser.add_argument( "--data_set", default="lipton/sentiment/orig/", type=str, required=False, help="")

    args = parser.parse_args()
    data_dir = args.data_dir
    dataset = args.data_set

    origd = os.path.join(data_dir ,dataset)

    orig_train = pd.read_table(origd+"train.tsv",sep="\t")   
    orig_dev = pd.read_table(origd+"dev.tsv",sep="\t")
    orig_test = pd.read_table(origd+"test.tsv",sep="\t")

    #made this folder prior to script
    train_out = os.path.join(origd,"bert_classifier_training/train.csv")
    dev_out = os.path.join(origd,"bert_classifier_training/dev.csv")
    test_out = os.path.join(origd,"bert_classifier_training/test.csv")

    create_classification_file(orig_train, train_out)
    create_classification_file(orig_test, test_out)
    create_classification_file(orig_dev, dev_out)

if __name__ == "__main__":
    main()
