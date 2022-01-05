import os
import nltk

root_dir = "C:/Users/Sriram/Documents/Research/Counterfactual_Local/Lipton/"

orig = "orig/"
new = "new/"
combined = "combined/"
paired = "paired/"

file1 = ['sentiment.train.0', 'sentiment.train.1',
         'sentiment.dev.0', 'sentiment.dev.1',
         'sentiment.test.0', 'sentiment.test.1']

for file_item in file1:
    with open(root_dir + combined + file_item, 'r', encoding='utf-8') as f:
        for item in f:
            item = item.strip()
            print(item)

# file2 = ['orig_reference.0', 'orig_reference.1', 'new_reference.0', 'new_reference.1']
# for file_item in file2:
#     with open(root_dir + combined + paired + file_item, 'r', encoding='utf-8') as f:
#         for instance in f:
#             text = instance.strip()
#             split_text = text.split('\t')
#             if(not len(split_text) == 2):
#                 print(text)
#                 print(len(split_text))
#                 quit(0)
#             item1 = split_text[0]
#             item2 = split_text[1]
#             print(item1 + "\n")
#             print(item2 + "\n\n")