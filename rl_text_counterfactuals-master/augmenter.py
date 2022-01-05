import os
os.environ["MODEL_DIR"] = 'aug_model'

#https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
with warnings.catch_warnings():
    # hide warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)

    # augmenter specific ( probably don't need nac or nas i or nafc)
    import nlpaug.augmenter.word as naw
    import time
    import pandas as pd

    #get embeddings at the word, sentence level
    import numpy as np
    import scipy
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')    #try cased at some point or other types of tokens?

    #sentiment analysis model prediction method
    import eval_with_tf192

    #get POS for new words in surrounding context
    from flair.data import Sentence
    from flair.models import SequenceTagger

    #for stopwords
    import string #.punctuation

# how to pass in stop words, punctuation
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

# NOW incorporate
# XXX 0. incorporate CF data and POS stuff
# 0.5 DO training with BERT CASED to see if it does better ( run in background )
# XXX 0.6 MAKE SELECTION OF WORD OCCUR HERE AND NOT IN SUBSITUTE METHOD

# 1. value function ( takes state and returns a single value estimate,
#   whereas for update call it takes state and G, and returns loss: 
#                self.loss = tf.squared_difference(self.value_estimate, self.target) 

# 2. policy function ( takes state and returns action_logits/probs/action … 
#        then on update it takes state/action/target and returns loss::  
# self.loss = -tf.log(self.picked_action_prob) * self.target   <— reward or cumulated)

# 3. how to form an initial representation of sentence and do i need to?
# do i need to actually account for POS?  

# state rep is word emb, sentence emb, pos(?), ner(?)
# action is substitute or not

# 3.5 clean up cf data to get rid of < br > html stuff

# 4. make gym environment ( use sentiment classifier for reward function )
# 5. try REINFORCE on CF data
# 6. try TD3 and SAC via spinup
# 7. analysis of learned policies via POS

def get_counterfactual_data():
    print("load cf data")
    cf_dataset = "data/all_cf_data.tsv"
    cf_train_df = pd.read_csv(cf_dataset, sep="\t")
    return cf_train_df

def get_pos_cf_info():
    final_pos = []
    with open('data/small_cf_pos_info.txt', 'r') as fp:
        for cnt, aline in enumerate(fp):
            final_pos.append(eval(aline))
    return final_pos

def calculate_distance_from_original(orig, newtext):
    # using this ACL 19 work:  https://github.com/UKPLab/sentence-transformers
    #both are expected to be strings if either is a list make it a string
    if type(orig) == 'list':
        orig = " ".join(orig)
    if type(newtext) == 'list':
        newtext = " ".join(newtext)

    sentences = [orig, newtext]
    embeddings = embedder.encode(sentences)
    orig_emb, new_emb = embeddings
    distances = scipy.spatial.distance.cdist([orig_emb], [new_emb], "cosine")[0]
    return distances

def clean_token(token):
    return token.translate(str.maketrans('', '', string.punctuation))
    
def main_cf(debug = False):
    st = time.time()

    #load cf sentences, label, and pos
    cf_data = get_counterfactual_data()
        
    # these preloaded pos info are off! TODO: do in real time instead
    cf_pos_info = get_pos_cf_info()

    sent_to_int = {"Negative":0, "Positive":1}
    cf_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ]
    cf_labels = [ sent_to_int[cf_data.iloc[a].Sentiment] for a in range(cf_data.shape[0]) ] 
    cf_pos = [ [a[1] for a in sent] for sent in cf_pos_info ]
    cf_tokens = [ [a[0] for a in sent] for sent in cf_pos_info ]

    if debug:
        print("Elapsed Time to Load Data:",time.time() - st)
    
    # load Contextual Word Embedder Augmenter
    aug = naw.ContextualWordEmbsAug( model_path='bert-base-uncased', action="substitute", stopwords=stopwords)

    # load POS tagger
    tagger = SequenceTagger.load('pos')

    for sid in range(len(cf_sentences)):
        ist = time.time()
        original_text = cf_sentences[sid]
        true_class = cf_labels[sid]
        #sent_pos = cf_pos[sid]
        orig_class =  eval_with_tf192.predict_sentiment_of_sentences([original_text],debug=False) 
        orig_class_label = orig_class['predicted_labels'][0]

        #use these tokens for subbing
        tokens = aug.split_text(original_text)[0].split(' ')  #this is 18, it includes ,

        tokens_text = " ".join(tokens)
        sentence = Sentence(tokens_text)   #only need this if you need tokenizer?
        tagger.predict(sentence)
        sent_pos = sentence.get_spans('pos')

        if debug:
            print(tokens,len(tokens))
            print(sent_pos,len(sent_pos))
        #print(cf_tokens[sid],len(cf_tokens[sid]))
        #import sys
        #sys.exit(0)
        #for i in range(len(sent_pos)):
        #       [i].to_dict()

        """
        cf_tokens = cf_tokens[sid]   #this is 14 and does
        cf_tokens_clean = [clean_token(toke.lower()) for toke in cf_tokens]

        #map from tokens to cleaned tokens so we can levarge the laters POS
        true_to_pos = [-1 for _ in range(len(tokens))]
        ti, lastfound = 0,0 
        while ti < len(tokens):
            ind = lastfound
            while ind < len(cf_tokens_clean):
                if cf_tokens_clean[ind] == tokens[ti]:
                    true_to_pos[ti] = ind
                    lastfound = ind
                    ind = len(cf_tokens_clean)
                ind += 1
            ti += 1

        if debug:
            print(tokens)
            print(cf_tokens_clean)
            print(true_to_pos)
        """


        augmented_text = tokens
       
        if debug:
            print("Original:", original_text,"\ntrue label",true_class,"\npred class/probs",orig_class['predicted_labels'],orig_class['predicted_probs'])
            print("ORIG Tokens:",tokens,"POS:",sent_pos,"CLEAN TOKENS",tokens)
            print(len(tokens),len(sent_pos)) #TODO what to do if these don't match up... maybe just CF_POS TOKENS?

        for i in range(len(tokens)):
            # there are 30522 words in the vocab 

            if tokens[i] not in stopwords:
                # PULL DECISION INTO HERE SO YOU CAN SEE VERIFY TOP CANDIDATES 
                # STOP USING THEIR TOKENIZER / WHAT TO DO ABOUT TEMPERATURE, DISTANCE, etc
                #cur_pos = sent_pos[true_to_pos[i]]
                cur_pos = sent_pos[i].to_dict()['type']
                cur_sent = ' '.join(augmented_text[i])
                if debug:
                    print("Considering subbing word ",augmented_text[i]," with POS",cur_pos)
                
                candidates = aug.substitute(augmented_text,[i])   
                candidates.sort(key = lambda x: x[1])
                candidates.reverse()
                if len(candidates) > 0:
                    #ENFORCE POS STUFF
                    can_tokens = tokens 
                    filtered_tokens, can_ps, can_dists = [],[],[]
                    can_words = [c[0] for c in candidates]   #0 word, 1 prob
                    if debug:
                        print("start with candidates",candidates)
                    for cw in range(len(can_words)):
                        can_tokens[i] = can_words[cw]
                        sent = ' '.join(can_tokens)
                        sentence = Sentence(sent) 
                        tagger.predict(sentence)
                        can_pos = sentence.get_spans('pos')[i].to_dict()
                        if can_pos['type'] != cur_pos:
                            if debug:
                                print("Filter out word (",can_words[cw],") that is of type",can_pos['type'])
                        else:
                            filtered_tokens.append(can_words[cw])
                            can_ps.append(candidates[cw][1])
                            cur_dist = calculate_distance_from_original(sent,cur_sent)
                            can_dists.append(cur_dist)

                    #post filtered
                    if len(can_ps) == 0:
                        #if no exact match found, use all
                        if debug:
                            print("No exact match found in candidates so keep all!")
                        filtered_tokens = can_words
                        can_ps = [c[1] for c in candidates]   #0 word, 1 prob

                    can_probs = [c / sum(can_ps) for c in can_ps]
                    if debug:
                        print("post filter", filtered_tokens)

                    #SHOULD I TAKE DISTANCE INTO ACCOUNT OR DO THAT VIA VALUE FUNCTION
                    can_ind =  np.random.choice(len(can_probs), p=can_probs)
                    substitute_word, prob =  filtered_tokens[can_ind], can_probs[can_ind]
                    tokens[i] = substitute_word
                    augmented_text = tokens

                    augmented_str = " ".join(augmented_text)
                    cur_dist = calculate_distance_from_original(original_text,augmented_str)
                    cur_class = eval_with_tf192.predict_sentiment_of_sentences([augmented_str])
                    if debug:
                        print(i,augmented_str,"DIST:",cur_dist,cur_class['predicted_labels'],cur_class['predicted_probs'])

                    cur_class_label = cur_class['predicted_labels'][0]

                    if cur_class_label != orig_class_label:
                        # if cur class not same as original stop at this CF
                        print("FOUND CF. ",orig_class_label, "--->",cur_class_label,"with dist",cur_dist)
                        break
            else:
                if debug:
                    print("SKIPPED stopword",tokens[i])
            

        print("Final Augmented Text:", augmented_text,"with predicted class",cur_class['predicted_labels'],cur_class['predicted_probs'])
        print("Example Elapsed Time:",time.time() - ist)  #3 seconds
        import sys
        sys.exit(0)



def main():
    #for initial debugging
    st = time.time()
    original_text = 'The quick brown fox jumps over the lazy dog .'
    #original_text = 'I really think the film Home Alone sucks.'
    orig_class =  eval_with_tf192.predict_sentiment_of_sentences([original_text],debug=False) 
    aug = naw.ContextualWordEmbsAug( model_path='bert-base-uncased', action="substitute")
    tokens = aug.split_text(original_text)[0].split(' ')
    augmented_text = original_text
   
    print("Original:", original_text,"with predicted class",orig_class['predicted_labels'],orig_class['predicted_probs'])
    for i in range(len(tokens)):
        augmented_text = aug.substitute_v2(augmented_text,[i])  #for use with substitute_v2
        cur_dist = calculate_distance_from_original(original_text,augmented_text)
        cur_class = eval_with_tf192.predict_sentiment_of_sentences([augmented_text])
        print(i,augmented_text,cur_dist,cur_class['predicted_labels'],cur_class['predicted_probs'])
        # there are 30522 words in the vocab 

    print("Final Augmented Text:", augmented_text,"with predicted class",cur_class)
    print("Elapsed Time:",time.time() - st)  #3 seconds

    # Original: The quick brown fox jumps over the lazy dog .
    # Augmented Text: a little blue figure crouched near my pet cat ;
    # Elapsed Time: 3.501749038696289
    """
    Original: The quick brown fox jumps over the lazy dog . with predicted class ['Positive'] [array([-1.2428893, -0.3404492], dtype=float32)]
    0 a quick brown fox jumps over the lazy dog . [0.00333105] ['Negative'] [array([-0.46336952, -0.99198484], dtype=float32)]
    1 a small brown fox jumps over the lazy dog . [0.04305757] ['Positive'] [array([-0.7377089 , -0.65048677], dtype=float32)]
    2 a small red fox jumps over the lazy dog . [0.06007847] ['Negative'] [array([-0.3988542, -1.1119668], dtype=float32)]
    3 a small red cat jumps over the lazy dog . [0.2529848] ['Negative'] [array([-0.36043292, -1.1952589 ], dtype=float32)]
    4 a small red cat hovered over the lazy dog . [0.34460213] ['Positive'] [array([-1.6089437 , -0.22326714], dtype=float32)]
    5 a small red cat hovered beside the lazy dog . [0.37006832] ['Positive'] [array([-1.5637324 , -0.23490398], dtype=float32)]
    6 a small red cat hovered beside a lazy dog . [0.38112253] ['Positive'] [array([-1.7945216, -0.1817701], dtype=float32)]
    7 a small red cat hovered beside a large dog . [0.38891887] ['Positive'] [array([-1.4591005, -0.2645455], dtype=float32)]
    8 a small red cat hovered beside a large boulder . [0.4393777] ['Positive'] [array([-0.97118723, -0.47583383], dtype=float32)]
    9 a small red cat hovered beside a large boulder ; [0.44164939] ['Positive'] [array([-1.4121194, -0.2792198], dtype=float32)]
    Final Augmented Text: a small red cat hovered beside a large boulder ; with predicted class {'sentences': ['a small red cat hovered beside a large boulder ;'], 'predicted_labels': ['Positive'], 'predicted_probs': [array([-1.4121194, -0.2792198], dtype=float32)]}

    # https://github.com/makcedward/nlpaug/blob/master/nlpaug/augmenter/word/context_word_embs.py

    # rl_cf_proj37/lib/python3.7/site-packages
            /nlpaug/augmenter/word/context_word_embs.py      #substitute method
            /nlpaug/augmenter/word/word_augmenter.py
            /nlpaug/model/lang_models/bert.py

    # https://arxiv.org/pdf/1805.06201.pdf
    """

def get_num_tokens():
    tagger = SequenceTagger.load('pos-fast')
    sentence = Sentence("The movie is bad")   
    tagger.predict(sentence)
    sent_pos = sentence.get_spans('pos')
    #print(sentence,dir(sentence))
    #print(tagger, dir(tagger))
    #print(sent_pos, dir(sent_pos))

    """
    'add_label', 'add_labels', 'add_token', 'clear_embeddings', 'convert_tag_scheme', 'embedding', 
    'get_embedding', 'get_label_names', 'get_language_code', 'get_spans', 'get_token', 'infer_space_after', 
    'labels', 'language_code', 'set_embedding', 'to', 'to_dict', 'to_original_text', 'to_plain_string', 
    'to_tagged_string', 'to_tokenized_string', 'tokenized', 'tokens'

    SequenceTagger(
      (embeddings): StackedEmbeddings(
        (list_embedding_0): FlairEmbeddings(
          (lm): LanguageModel( (drop): Dropout(p=0.25, inplace=False) (encoder): Embedding(275, 100) (rnn): LSTM(100, 1024) (decoder): Linear(in_features=1024, out_features=275, bias=True))
        )
        (list_embedding_1): FlairEmbeddings(
          (lm): LanguageModel( (drop): Dropout(p=0.25, inplace=False) (encoder): Embedding(275, 100) (rnn): LSTM(100, 1024) (decoder): Linear(in_features=1024, out_features=275, bias=True))
        )
      )
      (word_dropout): WordDropout(p=0.05)
      (locked_dropout): LockedDropout(p=0.5)
      (embedding2nn): Linear(in_features=2048, out_features=2048, bias=True)
      (rnn): LSTM(2048, 256, batch_first=True, bidirectional=True)
      (linear): Linear(in_features=512, out_features=20, bias=True)
    ) 
    add_module', 'apply', 'bidirectional', 'buffers', 'children', 'cpu', 'cuda', 'double', 'dump_patches', 
    'embedding2nn', 'embeddings', 'eval', 'evaluate', 'extra_repr', 'float', 'forward', 'forward_loss', 
    'get_transition_matrix', 'half', 'hidden_size', 'hidden_word', 'linear', 'load', 'load_state_dict', 
    'locked_dropout', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 
    'nlayers', 'parameters', 'pickle_module', 'predict', 'register_backward_hook', 'register_buffer', 
    'register_forward_hook', 'register_forward_pre_hook', 'register_parameter', 'relearn_embeddings', 'requires_grad_', 
    'rnn', 'rnn_layers', 'rnn_type', 'save', 'share_memory', 'state_dict', 'tag_dictionary', 'tag_type', 'tagset_size', 
    'to', 'train', 'train_initial_hidden_state', 'trained_epochs', 'training', 'transitions', 'type', 'use_crf', 
    'use_dropout', 'use_locked_dropout', 'use_rnn', 'use_word_dropout', 'word_dropout', 'zero_grad'

    [<DET-span (1): "The">, <NOUN-span (2): "movie">, <VERB-span (3): "is">, <ADJ-span (4): "bad">] 
     append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort'
    """
    #print(tagger.tag_dictionary, dir(tagger.tag_dictionary))   
    #<flair.data.Dictionary object at 0x7fb4cc063110>
    #'add_item', 'get_idx_for_item', 'get_idx_for_items', 'get_item_for_index', 'get_items', 'idx2item', 'item2idx', 'load', 'load_from_file', 'multi_label', 'save'
    #print(tagger.tag_dictionary.get_items())
    pos_types = ['<unk>', 'O', 'INTJ', 'PUNCT', 'VERB', 'PRON', 'NOUN', 'ADV', 'DET', 'ADJ', 'ADP', 'NUM', 'PROPN', 'CCONJ', 'PART', 'AUX', 'X', 'SYM', '<START>', '<STOP>']
    print(len(pos_types))
    
    

if __name__ == "__main__":
    #main()
    #main_cf(debug=True)
    get_num_tokens()
