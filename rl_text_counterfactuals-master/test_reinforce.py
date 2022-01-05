import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    import sys
    import numpy as np
    import gym
    import gym_counterfac
    import time
    #from matplotlib import pyplot as plt

    from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN
    import tensorflow as tf
    import pickle

# To frame it as a loss function it is enough to notice that gamma^t * G * grad(ln Pi(At|St)) = grad(gamma^t * G * ln(Pi(At|St)); 
# where REINFORCE basically maximizes the term in brackets on the right.

# https://web.stanford.edu/class/cs20si/lectures/CS20_intro_to_RL.pdf

def test_reinforce(with_baseline, num_episodes):
    env = gym.make("counterfac-v0")
    print("ENV INFO")
    print(env)        
    print(type(env))  
    print(dir(env))
    # <CounterFac<counterfac-v0>>
    # <class 'gym_counterfac.envs.counterfac_env.CounterFac'>
    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'action_space', 'aug', 'cf_all_labels', 'cf_all_sentences', 'cf_neg_sentences', 'cf_pos_sentences', 'close', 'counter', 'cur_class_label', 'current_iteration', 'current_pos', 'current_sentence', 'current_word', 'current_word_index', 'distance_from_original', 'done', 'human_cf', 'input_sentence', 'max_iterations', 'max_word_edits', 'metadata', 'minimum_word_edits_needed', 'next_word', 'num_edits', 'num_non_stopwords', 'num_words_edited', 'observation_space', 'original_label', 'percent_changed_needed', 'render', 'reset', 'reward', 'reward_range', 'seed', 'sent_pos', 'sent_to_int', 'spec', 'state', 'step', 'subword', 'tagger', 'tokens', 'tokens_text', 'unwrapped', 'words_edited']

    objs = ['action_space', 'add', 'check', 'close', 'counter', 'done', 'metadata', 'observation_space', 'render', 'reset', 'reward', 'reward_range', 'seed', 'spec', 'state', 'step', 'unwrapped']
    print('action_space',env.action_space,type(env.action_space), env.action_space.n)
    print('observation_space',env.observation_space,type(env.observation_space),env.observation_space.n)
    print('meta_data',env.metadata,type(env.metadata))
    print('reward_range',env.reward_range,type(env.reward_range))
    print('state',env.state) 
   
    #discount rate, and learning rate
    gamma = 1.
    alpha = 3e-4

    if 'tensorflow' in sys.modules:
        #import tensorflow as tf
        tf.reset_default_graph()

    #print(env.observation_space.shape)
    #env.observation_space.shape[0],
    state_dims = 1556  #current word is emb of 768, context is emb of 768 and pos is one hot vec of 20
    #pos_types = ['<unk>', 'O', 'INTJ', 'PUNCT', 'VERB', 'PRON', 'NOUN', 'ADV', 'DET', 'ADJ', 'ADP', 'NUM', 'PROPN', 'CCONJ', 'PART', 'AUX', 'X', 'SYM', '<START>', '<STOP>']

    pi = PiApproximationWithNN(
        env.observation_space.n,
        env.action_space.n,
        alpha)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.n,
            alpha)
    else:
        B = Baseline(0.)

    return REINFORCE(env,gamma,num_episodes,pi,B)   #do 15 episodes only

if __name__ == "__main__":
    fileout = ""
    # EXPECTED USAGE:  python test_reinforce.py fileout_prefix num_episodes reinforce_type
    if len(sys.argv) > 1:
        print("passed in fileout_prefix and num_episodes to run of: " , str(sys.argv))
        fileout = sys.argv[1]
        num_episodes = int(sys.argv[2])
        todo = sys.argv[3]                   #todo is either 'both', 'withb', 'nobase'
    else:
        #unless passed in default to 3 episodes for debuging
        num_episodes = 3
        todo = 'both'

    start_time = time.time()
    
    # Test REINFORCE without baseline
    if todo == 'both' or todo == 'nobase':
        training_progress = test_reinforce(with_baseline=False, num_episodes=num_episodes)
        without_baseline = np.mean(training_progress["episode_rewards"],axis=0)
        print("Without baseline", without_baseline)
        if fileout != "":
            without_fname = "experiments/"+fileout + "_without_results.p"
            pickle.dump( training_progress, open( without_fname, "wb" ) )
    
    # Test REINFORCE with baseline
    if todo == 'both' or todo == 'withb':
        with_baseline = []
        training_progress = test_reinforce(with_baseline=True, num_episodes=num_episodes)
        with_baseline.append(training_progress["episode_rewards"])
        with_baseline = np.mean(with_baseline,axis=0)
        print("With baseline", with_baseline)
        if fileout != "":
            with_fname = "experiments/" + fileout + "_withbaseline_results.p"
            pickle.dump( training_progress, open( with_fname, "wb" ) )

    end_time = time.time()
    print("Elapsed time: ",end_time - start_time )
