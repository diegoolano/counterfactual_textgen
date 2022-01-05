import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
os.environ["MODEL_DIR"] = 'aug_model'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    #for augmenter
    import nlpaug.augmenter.word as naw
    import numpy as np
    import pandas as pd
    import math

    # for distance calculation
    import scipy
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')    

    # sentiment analysis model prediction method
    import eval_with_tf192

    # pos tagger
    from flair.data import Sentence
    from flair.models import SequenceTagger

    from scipy.special import softmax

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']


POSTYPES = ['<unk>', 'O', 'INTJ', 'PUNCT', 'VERB', 'PRON', 'NOUN', 'ADV', 'DET', 'ADJ', 'ADP', 'NUM', 'PROPN', 'CCONJ', 'PART', 'AUX', 'X', 'SYM', '<START>', '<STOP>']

def get_counterfactual_data():
    cf_dataset = "/home/diego/rl_proj_cf/data/all_cf_data.tsv"
    cf_train_df = pd.read_csv(cf_dataset, sep="\t")
    return cf_train_df

def calculate_distance_between_sentences(orig, newtext):
    # using this ACL 19 work:  https://github.com/UKPLab/sentence-transformers
    #both are expected to be strings if either is a list make it a string
    if type(orig) == 'list':
        orig = " ".join(orig)
    if type(newtext) == 'list':
        newtext = " ".join(newtext)

    sentences = [orig, newtext]
    embeddings = embedder.encode(sentences)
    orig_emb, new_emb = embeddings

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    #Computes the cosine distance between vectors u and v:  1 - cosine_sim(u,v)
    distance = scipy.spatial.distance.cdist([orig_emb], [new_emb], "cosine")[0]
    return distance

class CounterFac(gym.Env):
    metadata = {'render.modes': ['human']}
    # https://gym.openai.com/docs/#environments

    def __init__(self):
        self.sent_to_int = {"Negative":0, "Positive":1}
        cf_data = get_counterfactual_data()

        # FOR NOW ONLY ALLOW CHANGES OF NEGATIVE TO POSITIVE
        self.cf_all_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ]
        self.cf_all_labels = [ self.sent_to_int[cf_data.iloc[a].Sentiment] for a in range(cf_data.shape[0]) ] 
        self.cf_neg_sentences, self.cf_pos_sentences = [],[]
        for i in range(len(self.cf_all_sentences)):
            if self.cf_all_labels[i] == 0:
                self.cf_neg_sentences.append(self.cf_all_sentences[i])
            else:
                self.cf_pos_sentences.append(self.cf_all_sentences[i])

        cf_neg_meta = [ (c,len(c)) for c in self.cf_neg_sentences]    
        cf_neg_meta.sort(key = lambda x: x[1])
        self.cf_neg_sentences = [ c[0] for c in cf_neg_meta]
        
        #print("NUM OF SENTENCES AND LEN OF SHORTEST SENTENCES VS LARGEST", len(self.cf_neg_sentences), len(self.cf_neg_sentences[0]), len(self.cf_neg_sentences[-1]))
        # 2440 79 2051
        #import sys
        #sys.exit(0)

        cf_pos_meta = [ (c,len(c)) for c in self.cf_pos_sentences]    
        cf_pos_meta.sort(key = lambda x: x[1])
        self.cf_pos_sentences = [ c[0] for c in cf_pos_meta]   #THIS IS NOT RIGHT.. it needs to be based on the map from the be

        self.aug = naw.ContextualWordEmbsAug( model_path='bert-base-uncased', action="substitute", stopwords=STOPWORDS)

        # load POS tagger
        self.tagger = SequenceTagger.load('pos-fast')

        # HYPERPARAMETERS TO SET RULES OF GAME
        # amount of sentence which must be changed before we allow it to be considered a CF
        self.percent_changed_needed = .20

        # max number of edits that can be made before episode is done
        self.max_word_edit_percent = 1   #this actually is way too restrictive in V1 game using .5 , but probably any value really as we'd like to be able to iterate

        # max number of iterations through the sentence that can be done before episode is done
        self.max_iterations = 5
        self.row_id = 0
        self.reset(True)   #just set true to initialize first time

    def reset(self, init=False):
        # state = env.reset()

        # get counterfactual from dataset at random
        #cf_ind =  np.random.choice(len(self.cf_neg_sentences))
    
        # get cf from dataset starting with smallest
        cf_ind = self.row_id
        self.row_id +=1 
        self.input_sentence = self.cf_neg_sentences[cf_ind]
        self.human_cf = self.cf_pos_sentences[cf_ind]
        self.original_label = 0 #self.cf_labels[cf_ind]
    
        self.current_sentence = self.cf_neg_sentences[cf_ind]
        self.current_word_index = 0
        self.current_iteration = 0
        self.cur_class_label = 0
        self.num_edits = 0
        self.num_words_edited = 0
        self.words_edited = []

        self.tokens = self.aug.split_text(self.input_sentence)[0].split(' ')  #this is 18, it includes ,
        self.num_non_stopwords = 0

        #to prevent duplicate words mostly
        self.prior_words_used = []
        for i in  range(len(self.tokens)):
            if self.tokens[i] not in STOPWORDS:
                self.num_non_stopwords += 1
                if self.tokens[i] not in self.prior_words_used:
                    self.prior_words_used.append(self.tokens[i])

        #print(self.input_sentence)
        #print(self.prior_words_used)

        self.minimum_word_edits_needed = min(math.ceil(self.num_non_stopwords * self.percent_changed_needed),15)
        if self.minimum_word_edits_needed < 5:
            self.minimum_word_edits_needed = 5
            
        #this is too restritive so for now let be 3 * num of stopwords
        #self.max_word_edits = min(math.ceil(self.num_non_stopwords * self.max_word_edit_percent),50) 
        self.max_word_edits = self.num_non_stopwords * self.max_iterations
        self.max_counter_before_stop = self.num_non_stopwords * (self.max_iterations + 3)   #to prevent runaway NOCHANGEs

        self.softmaxed = 0
        
        #only need this if you need tokenizer
        self.tokens_text = " ".join(self.tokens)
        sentence = Sentence(self.tokens_text)   
        self.tagger.predict(sentence)
        self.sent_pos = sentence.get_spans('pos')

        self.next_word()
        
        self.make_embedded_version()

        self.state = [self.current_word, self.current_sentence, self.current_pos, self.cur_emb]      
        self.counter = 0
        self.done = 0
        self.distance_from_original = 0
        self.reward = 0
        self.action_space = spaces.Discrete(2)      # substitute or don't
        
        #self.observation_space = spaces.Box(1560)   # state observation will be [word embedding, sentence embedding, pos ]  #i think this is 768 + 768 + one hot encoded pos (20) 
        self.observation_space = spaces.Discrete(1556)   
        if init == False:
            return self.state

    def make_embedded_version(self):
        debug = False
        embeddings = embedder.encode([self.current_word, self.current_sentence])
        word_emb, sent_emb = embeddings
        pos_emb = np.zeros(len(POSTYPES))
        pos_emb[POSTYPES.index(self.current_pos)] = 1
        if debug:
            print(type(word_emb),type(sent_emb),type(pos_emb))
            print(word_emb.shape, sent_emb.shape, pos_emb.shape)
        self.cur_emb = np.concatenate((word_emb, sent_emb, pos_emb), axis=0)
        

    def next_word(self):
        # state will be ( cur_word, sentence, pos ) 
        while self.current_word_index < len(self.tokens):
            if self.tokens[self.current_word_index] not in STOPWORDS:
                break
            else:
                self.current_word_index += 1

        if self.current_word_index == len(self.tokens):
            # if you reached the end of the sentence start again from beginning and bump up iteration
            self.current_word_index = 0
            self.current_word = self.tokens[self.current_word_index]            
            self.current_iteration += 1
            self.next_word()
        else:
            self.current_word = self.tokens[self.current_word_index]            
            self.current_pos = self.sent_pos[self.current_word_index].to_dict()['type']

    def filter_candidates(self, candidates, by_pos=True, by_prior_use=True):
        # filter based on POS mismatch and/or
        # filter out words that are already in sentence ( to prevent boring, boring, boring)
        debug = True
        can_tokens = self.tokens.copy() 
        filtered_tokens, all_pos, filt_pos, can_ps, can_dists = [],[],[],[],[]
        can_words = [c[0] for c in candidates]   #0 word, 1 prob

        if debug:
            print("start with candidates",candidates)

        for cw in range(len(can_words)):
            can_tokens[cw] = can_words[cw]
            sent = ' '.join(can_tokens)
            sentence = Sentence(sent) 
            self.tagger.predict(sentence)
            can_pos = sentence.get_spans('pos')[cw].to_dict()
            all_pos.append(can_pos)

            filter_out_word = False 

            """ #this filters out candidates from being subbed in, but really we want to avoid subbing these words in the first place
            if can_pos['type'] in ['PRON', 'PROPN']:
                filter_out_word = True
                if debug:
                    print("POS PRON / PROPN so filter out word (",can_words[cw],")")
            """
            if by_pos == True and by_prior_use == True and (can_pos['type'] != self.current_pos or can_words[cw] in self.prior_words_used):
                filter_out_word = True
                if debug:
                    if can_pos['type'] != self.current_pos:
                        print("POS MISMTACH Filter out word (",can_words[cw],") that is of type",can_pos['type'])
                    if can_words[cw] in self.prior_words_used:
                        print("CANDIDATE PRIORLY USED Filter out word (",can_words[cw],")")
            elif by_pos == True and can_pos['type'] != self.current_pos: 
                filter_out_word = True
                if debug:
                    print("POS MISMTACH Filter out word (",can_words[cw],") that is of type",can_pos['type'])
            elif by_pos == True and can_pos['type'] != self.current_pos: 
                filter_out_word = True
                if debug:
                    print("POS MISMTACH Filter out word (",can_words[cw],") that is of type",can_pos['type'])

            if filter_out_word == False:
                filtered_tokens.append(can_words[cw])
                filt_pos.append(can_pos['type'])
                can_ps.append(candidates[cw][1])
                cur_dist = calculate_distance_between_sentences(sent,self.current_sentence)
                can_dists.append(cur_dist)

        #post filtered
        if len(can_ps) == 0:
            #if no exact match found, use all
            if debug:
                print("No exact match found in candidates so keep all!")
            filtered_tokens = can_words
            #filt_pos = all_pos
            filt_pos = [ a['type'] for a in all_pos]
            can_ps = [c[1] for c in candidates]   #0 word, 1 prob

        can_probs = [c / sum(can_ps) for c in can_ps]
        if debug:
            print("post filter", filtered_tokens, filt_pos, can_ps)

        return can_probs, filtered_tokens, filt_pos

    def subword(self):
        debug = True
        if debug:
            print("Substitute index",self.current_word_index, self.current_word, self.current_pos, "in", self.tokens) 

        if self.current_pos in ['PRON', 'PROPN']:
            if debug:
                print("POS PRON / PROPN so filter out word (",self.current_word," , ",self.current_pos,")")
            return -1

        #right now 5 candidates are returned but we could make it more and/or account for distance
        candidates = self.aug.substitute(self.tokens.copy(),[self.current_word_index])   

        if debug:
            print("")
            print("Post candidate: Substitute index",self.current_word_index, self.current_word, self.current_pos, "in", self.tokens) 

        if len(candidates) == 1:
            if candidates[0][0] == "INVALID WORD FOR SUB":
                return -1
                    
        candidates.sort(key = lambda x: x[1])
        candidates.reverse()

        if len(candidates) > 0:
            # Now filter
            can_probs, filtered_tokens, filt_pos = self.filter_candidates(candidates, by_pos=True, by_prior_use=True)

            #TODO: SHOULD I TAKE DISTANCE INTO ACCOUNT OR DO THAT VIA VALUE FUNCTION
            can_ind =  np.random.choice(len(can_probs), p=can_probs)
    
            if debug:
                print("Can ind",can_ind, filtered_tokens[can_ind], can_probs[can_ind], filt_pos[can_ind])
                print("Current word index/word, tokens[current_index]", self.current_word_index, self.current_word, self.tokens[self.current_word_index])
                print("Current sentence", self.current_sentence)
                print("Curent tokens:", self.tokens)

            substitute_word, prob, sub_pos =  filtered_tokens[can_ind], can_probs[can_ind], filt_pos[can_ind]
            self.tokens[self.current_word_index] = substitute_word
            self.changed_to = [substitute_word, sub_pos]

            if substitute_word in self.prior_words_used:
                print("WARNING.  changed current_word:",self.current_word," to ", substitute_word," which is on the priorly used list.. this should be late restort")
            else:
                self.prior_words_used.append(substitute_word)

            #TODO: NEXT fix spaces " , "
            self.current_sentence = " ".join(self.tokens)   
            self.current_sentence = self.current_sentence.replace(" ,",",").replace(" .",".").replace(" ' ", "'").replace(" ?","?") 

            if debug:
                print("----")
                print("Post subsitute: current word index/word, tokens[current_index]", self.current_word_index, self.current_word, self.tokens[self.current_word_index])
                print("changed_to: ",self.changed_to)
                print("Post Current sentence", self.current_sentence)

            if self.current_word_index not in self.words_edited:
                self.words_edited.append(self.current_word_index)
                self.num_words_edited += 1

            if debug:
                print("Num edits is ",self.num_edits)

            self.num_edits += 1

            if debug:
                print("Num edits is ",self.num_edits)
                print("Current word index is ",self.current_word_index)

            # need to bump up index since a sub was found
            self.current_word_index += 1
            self.next_word()   

            if debug:
                print("Post call to next current word index/word, tokens[current_index]", self.current_word_index, self.current_word, self.tokens[self.current_word_index])
            
            #TODO: should reward be between subsequent edits or between current and original str
            # For now do current and original sentence
            #cur_dist = calculate_distance_from_original(self.input_sentence,self.current_sentence)
            cur_dist = calculate_distance_between_sentences(self.input_sentence, self.current_sentence)

            # how we enforce that at least a certain amount of edits have occured 
            # before allowing to check if label has changed
            if (self.num_edits > self.minimum_word_edits_needed):
                cur_class = eval_with_tf192.predict_sentiment_of_sentences([self.current_sentence])
                self.softmaxed = softmax(cur_class['predicted_probs'])
                if debug:
                    print("RUNNING PREDICTION ON", self.current_sentence, "DIST:",cur_dist,cur_class['predicted_labels'],cur_class['predicted_probs'], self.softmaxed)

                cur_class_label = cur_class['predicted_labels'][0]
                if cur_class_label != "Negative":  
                    # if cur class not same as original stop at this CF
                    if debug == debug:
                        print("DONE FOUND CF!! with dist",cur_dist)
                    self.cur_class_label = 1
                    self.done = 1
                elif (self.current_iteration > self.max_iterations) or (self.num_edits > self.max_word_edits):
                    if debug == debug:
                        print("DONE BECAUSE MAX_ITERATIONS REACHED OR MAX EDITS REACHED")
                        print("Current Iteration > max iterations", self.current_iteration,"versus", self.max_iterations, "OR num edits > max word edits", self.num_edits,"vs",self.max_word_edits)
                    self.cur_class_label = 0
                    self.done = 1
                elif self.counter > self.max_counter_before_stop: 
                    if debug == debug:
                        print("DONE BECAUSE MAX_COUNTER REACHED ")
                        print("Current counter > max counter", self.counter,"versus", self.max_counter_before_stop)
                    self.cur_class_label = 0
                    self.done = 1
                else:
                    nothing = 1
                    #print("Current Iteration < max iterations", self.current_iteration,"versus", self.max_iterations, "and num edits < max word edits", self.num_edits,"vs",self.max_word_edits)
            else:
                self.softmaxed = 0

        return cur_dist

    def skipword(self):
        debug = False
        if debug:
            print("SKIP WORD",self.current_word, self.current_word_index)
        # skip word so reward zero, go to next word, make embs and update state
        self.current_word_index += 1

        self.reward = 0
        self.next_word()                 #sets current_word and current_pos
        self.make_embedded_version()     #sets cur_emb  
        if debug:
            print("NEXT WORD IS",self.current_word, self.current_word_index)
        self.state = [self.current_word, self.current_sentence, self.current_pos, self.cur_emb]      #reuse current sentence from prior step since it didn't change

    def step(self, action):
        # action=O means SUBWORD, action=1 means SKIP

        debug = False
        # next_state, reward, done, _ = env.step(action)
        self.counter += 1
        print("STEP:",self.counter, "out of", self.max_counter_before_stop)
        dist_multiplier = 10  #was 100, but make it smaller.. 
        
        prior_word = [self.current_word, self.current_pos]
        if action == 0:
            if debug:
                print("SUB WORD ",prior_word," (now at ",self.num_edits," edits out of ",self.minimum_word_edits_needed,"minimum needed, and ",self.max_word_edits," max possible)")
            # substitute current word and get distance 
            # TODO: do i need to allow policy to learn which word to substitute with
            cur_dist = self.subword()

            if cur_dist == -1:
                self.skipword()
            else:
                #if not skipped, update distance
                self.distance_from_original = cur_dist

            self.make_embedded_version()
        
            #set current state
            self.state = [self.current_word, self.current_sentence, self.current_pos, self.cur_emb]      

            # USE INFO to pass ( [Prior Word,Prior Pos] -> [Sub Word, Sub Pos] )
            goal_reached = 0
            
            # For Rewards, 
            # The idea is we want the environment to: 
            #  - reward moving a small distance and 
            #  - penalize big jumps and         
            #  - penalize for each step
            #  - give a big reward when a CF is found ( but still accounting for distance and coherency)
            # distance is between 0 and 1 with zero being no move and 1 being the farthest

            if self.done == 1:
                if self.cur_class_label == 1:
                    #found counterfactual
                    self.reward = 100 - (dist_multiplier * cur_dist)
                    goal_reached = 1
                else:
                    #done cause maxed iterations or word edits,  Penalize Heavily (regardless of distance?)
                    self.reward = -100 + (1/cur_dist) 
                info = [ prior_word, self.changed_to, self.distance_from_original, self.softmaxed, goal_reached ]
                return [self.state, self.reward, self.done, info]   
            else:
                # not done so calculate reward
                #self.reward = -1 - ( 100 * cur_dist)
                # this was + with dist_mult=100 and did okay  
        
                #self.reward = (1 / cur_dist )   #encourage moves less than
                self.reward = min(4.5,( 1 / np.exp(math.log(cur_dist,10))) - 2) * 3  #so you can get max 10 reward, min -5  where breakeven is distance of .4 


                #HERE ITS HARD TO KNOW WHETHER ITS BETTER TO HAVE REWARDS which are based on prior step or just based on distance from input.  for now keep it as from input.

                #or maybe consider a joint thing which takens into account distance from the original input and distance from the prior one
                #ie, you can be far from prior step as long as you aren't too far from original input

                info = [ prior_word, self.changed_to, self.distance_from_original, self.softmaxed, goal_reached ]
        else:
            # SKIP WORD
            info = [ prior_word, prior_word, self.distance_from_original, 0, 0] 
            if self.counter > self.max_counter_before_stop: 
                if debug == debug:
                    print("DONE BECAUSE MAX_COUNTER REACHED ")
                    print("Current counter > max counter", self.counter,"versus", self.max_counter_before_stop)
                self.cur_class_label = 0
                self.done = 1
            else:
                # skip word and give reward 0 
                self.skipword()

        return [self.state, self.reward, self.done, info]

    def render(self):
        print("##############")
        print("Original Sentence:",self.input_sentence) 
        print("Current Sentence:",self.current_sentence)
        print("Current Word and Index",self.current_word,self.current_word_index)
        print("Current Distance from Original:",self.distance_from_original)
        print("Current Iteration through sentence: ",self.current_iteration)
        print("Total num edits:",self.num_edits)
        print("Unique words edited:",self.num_words_edited)
        print("##############")
