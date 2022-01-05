import csv
import logging
import os
import random
import sys
import numpy as np
import pandas as pd

#before doing the following I need to create sentiment_train_0 , sentiment_train_1 , etc versions of our training data
#TODO: is \t the correct delimiter we desire?

double_quotes = chr(34)
single_quote = chr(39)
space = " "

# def create_classification_file(input_df, output_file_path):
#   with open(output_file_path, "w", encoding='utf-8', newline='') as out_fp:
#     writer = csv.writer(out_fp, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
#     for i in range(input_df.shape[0]):
#       line = input_df.Text.values[i].replace('<br />', ' ')
#       line = line.strip(space + double_quotes + single_quote) #remove leading and trailing whitespace and quotes
#       #if(i%150==1):
#         #print(line+"\n")
#       writer.writerow([line])

def create_classification_file(input_df, output_file_path):
  with open(output_file_path, "w", encoding='utf-8', newline='') as out_fp:
    for i in range(input_df.shape[0]):
      line = input_df.Text.values[i].replace('<br />', ' ')
      line = line.strip(space + double_quotes + single_quote) #remove leading and trailing whitespace and quotes
      #if(i%150==1):
        #print(line+"\n")
      out_fp.write(line + "\n")
            
root_dir = "C:/Users/Sriram/Documents/Research/Counterfactual_Local/Lipton/"

orig = "orig/"
new = "new/"
combined = "combined/"

dev = "dev.tsv"
test = "test.tsv"
train = "train.tsv"

pos = "Positive"
neg = "Negative" 

all_data = [orig, new, combined]

for data in all_data:

  train_df = pd.read_table(root_dir + data + train, sep="\t")
  dev_df = pd.read_table(root_dir + data + dev, sep="\t")
  test_df = pd.read_table(root_dir + data + test, sep="\t")
  
  train_neg = train_df[train_df.Sentiment == neg][["Text"]]
  train_pos = train_df[train_df.Sentiment == pos][["Text"]]
  dev_neg = dev_df[dev_df.Sentiment == neg][["Text"]]
  dev_pos = dev_df[dev_df.Sentiment == pos][["Text"]]
  test_neg = test_df[test_df.Sentiment == neg][["Text"]]
  test_pos = test_df[test_df.Sentiment == pos][["Text"]]

  print(train_neg.size)
  print(train_pos.size)
  print(dev_neg.size)
  print(dev_pos.size)
  print(test_neg.size)
  print(test_pos.size)

  create_classification_file(train_neg, root_dir + data + "sentiment.train.0" )
  create_classification_file(train_pos, root_dir + data + "sentiment.train.1" )
  create_classification_file(dev_neg, root_dir + data + "sentiment.dev.0" )
  create_classification_file(dev_pos, root_dir + data + "sentiment.dev.1" )
  create_classification_file(test_neg, root_dir + data + "sentiment.test.0" )
  create_classification_file(test_pos, root_dir + data + "sentiment.test.1" )
  
