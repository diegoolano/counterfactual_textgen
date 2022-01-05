import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    from flair.data import Sentence
    from flair.models import SequenceTagger

    import pandas as pd
    import time


def get_counterfactual_data():
    print("load cf data")
    cf_dataset = "data/all_cf_data.tsv"
    cf_train_df = pd.read_csv(cf_dataset, sep="\t")
    return cf_train_df

def main():
    start_time = time.time()

    # load the POS tagger
    tagger = SequenceTagger.load('pos')

    cf_data = get_counterfactual_data()
    pred_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ]

    final_pos = []
    for sent in pred_sentences:
        sentence = Sentence(sent)
        tagger.predict(sentence)

        pos_info = []
        for entity in sentence.get_spans('pos'):
            pos_info.append(entity.to_dict()) 
        final_pos.append(pos_info)

    print("Elapsed Time:",time.time() - start_time)
    with open('cf_pos_info.txt', 'w') as f:
        for item in final_pos:
            f.write("%s\n" % item)

def convert_large_to_small_output():
    start_time = time.time()
    final_pos = []
    with open('cf_pos_info.txt', 'r') as fp:
        for cnt, aline in enumerate(fp):
            aline = eval(aline)
            small_line = [ (aline[i]['text'],aline[i]['type']) for i in range(len(aline)) ]
            final_pos.append(small_line)

    with open('small_cf_pos_info.txt', 'w') as f:
        for item in final_pos:
            f.write("%s\n" % item)
    print("Elapsed Time:",time.time() - start_time)
    #Elapsed Time: 9.629307985305786
    

if __name__ == "__main__":
    #main()
    convert_large_to_small_output()
