import argparse
import glob
import os
import csv

"""
# ACL data ( 50000 reviews:  25k train / 25k test  , both evenly divided between pos/neg  ( 12.5k each )
# data/aclImdb  -->  imdbEr.txt  imdb.vocab  README  test  train

#    aclImdb/train/ --> labeledBow.feat  neg  pos  unsup  unsupBow.feat  urls_neg.txt  urls_pos.txt  urls_unsup.txt
#    aclImdb/train/pos/  0_9.txt  to    12499_7.txt    #each of the 12500 files is a review itself  ( neg also has 12500 )

# for our purposes use half of test pos/neg for dev ( so 50% train, 25% dev, 25% test ) 

# Whereas what it TDRG expects looks like this
diego@microdeep:~/spr20_cf_gen/TDRG/data/yelp/bert_classifier_training$ head train.csv 
i was sadly mistaken .  0
so on to the hoagies , the italian is general run of the mill . 0
minimal meat and a ton of shredded lettuce .    0

# so we want to make liptons look like the yelp output so we can feed this into run_classifier.py

"""
def read_in_from_and_write_to_with_label(input_path, writer, label, data_type=0):
    print("calling read_in with", input_path, label, data_type )
    count = 0
    if data_type == 0:
        # handle train
        print("TRAIN")
        for filename in glob.glob(os.path.join(input_path, '*.txt')):
           with open(filename, 'r') as f:
                line = f.read()
                writer.writerow([line.strip(),label])

    elif data_type == 1:
        # handle dev
        print("DEV")
        for filename in glob.glob(os.path.join(input_path, '*.txt')):
            if count < 6250:
               with open(filename, 'r') as f:
                    line = f.read()
                    writer.writerow([line.strip(),label])
            count += 1
        print(count)
    else:
        # handle test
        print("TEST")
        for filename in glob.glob(os.path.join(input_path, '*.txt')):
            if count >= 6250:
               with open(filename, 'r') as f:
                    line = f.read()
                    writer.writerow([line.strip(),label])
            count += 1
        print(count)


def create_classification_file(input_path, output_file_path, data_type = 0):
    """
    Create a tsv file combining training data for BERT classification training.
    """
    with open(output_file_path, "w") as out_fp:
        writer = csv.writer(out_fp, delimiter="\t") #, quoting = csv.QUOTE_NONE)
        train_neg_path = os.path.join(input_path,"neg")
        read_in_from_and_write_to_with_label(train_neg_path, writer, label=0, data_type=data_type)

        train_pos_path = os.path.join(input_path,"pos")
        read_in_from_and_write_to_with_label(train_pos_path, writer, label=1, data_type=data_type)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument( "--data_dir", default="../data/", type=str, required=False, help="")
    parser.add_argument( "--data_set", default="aclImdb/", type=str, required=False, help="")
    args = parser.parse_args()
    data_dir = args.data_dir
    dataset = args.data_set
    origd = os.path.join(data_dir ,dataset)

    #made this folder prior to script
    train_out = os.path.join(origd,"bert_classifier_training/train.csv")
    dev_out = os.path.join(origd,"bert_classifier_training/dev.csv")
    test_out = os.path.join(origd,"bert_classifier_training/test.csv")

    create_classification_file(os.path.join(origd,"train"), train_out, 0)
    create_classification_file(os.path.join(origd,"test"), dev_out, 1)
    create_classification_file(os.path.join(origd,"test"), test_out, 2)

    """
    diego@microdeep:~/spr20_cf_gen/TDRG/data/aclImdb/bert_classifier_training$ cat test.csv | wc -l
    12500
    diego@microdeep:~/spr20_cf_gen/TDRG/data/aclImdb/bert_classifier_training$ cat dev.csv | wc -l
    12500
    diego@microdeep:~/spr20_cf_gen/TDRG/data/aclImdb/bert_classifier_training$ cat train.csv | wc -l
    25000
    """
if __name__ == "__main__":
    main()
