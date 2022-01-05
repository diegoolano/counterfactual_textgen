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

# def create_classification_file(input_df1, input_df2, output_file_path):
#   with open(output_file_path, "w", encoding='utf-8', newline='') as out_fp:
#     writer = csv.writer(out_fp, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
#     for i in range(input_df1.shape[0]):
#       line1 = input_df1.Text.values[i].replace('<br />', ' ')
#       line2 = input_df2.Text.values[i].replace('<br />', ' ')

#       line1 = line1.strip(space + double_quotes + single_quote)
#       line2 = line2.strip(space + double_quotes + single_quote)

#       writer.writerow([line1 + "\t" + line2])

def create_classification_file(input_df1, input_df2, output_file_path):
  with open(output_file_path, "w", encoding='utf-8', newline='') as out_fp:
    for i in range(input_df1.shape[0]):
      line1 = input_df1.Text.values[i].replace('<br />', ' ')
      line2 = input_df2.Text.values[i].replace('<br />', ' ')

      line1 = line1.strip(space + double_quotes + single_quote)
      line2 = line2.strip(space + double_quotes + single_quote)

      out_fp.write(line1 + "\t" + line2 + "\n")
            
root_dir = "C:/Users/Sriram/Documents/Research/Counterfactual_Local/Lipton/combined/paired/"

test = "test_paired.tsv"

pos = "Positive"
neg = "Negative" 

num_samples = 488

test_df = pd.read_table(root_dir + test, sep="\t")

orig_pos = pd.DataFrame(columns=test_df.columns)
orig_neg = pd.DataFrame(columns=test_df.columns)
new_pos = pd.DataFrame(columns=test_df.columns)
new_neg = pd.DataFrame(columns=test_df.columns)

for i in range(num_samples):
  sample = test_df.iloc[i*2]
  
  if sample["Sentiment"] == neg:
    orig_neg = orig_neg.append(sample, ignore_index = True)
    new_pos = new_pos.append(test_df.iloc[i*2 + 1], ignore_index = True)
  else:
    orig_pos = orig_pos.append(sample, ignore_index = True)
    new_neg = new_neg.append(test_df.iloc[i*2 + 1], ignore_index = True)

print(orig_pos.shape)
print(orig_neg.shape)
print(new_pos.shape)
print(new_neg.shape)

orig_pos = orig_pos[["Text"]]
orig_neg = orig_neg[["Text"]]
new_pos = new_pos[["Text"]]
new_neg = new_neg[["Text"]]

print(orig_pos.shape)
print(orig_neg.shape)
print(new_pos.shape)
print(new_neg.shape)

create_classification_file(orig_pos, new_neg, root_dir + "orig_reference.1")
create_classification_file(orig_neg, new_pos, root_dir + "orig_reference.0")
create_classification_file(new_pos, orig_neg, root_dir + "new_reference.1")
create_classification_file(new_neg, orig_pos, root_dir + "new_reference.0")