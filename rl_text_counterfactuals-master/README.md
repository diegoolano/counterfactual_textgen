
## To install, create a new virtual env with Python 3.7, activate it and install requirements via
    virtualenv --python=/home/diego/anaconda3/envs/py37/bin/python rl_cf_proj37
    source rl_cf_proj37/bin/activate
    pip install --requirement requirements.txt

## Then pip install custom open AI gym environment
    cd envs/counterfacs/gym-counterfac/gym_counterfac/
    pip install -e .

## Substantial changes were made to the nlpaug package so after installing requirements, 
    cp -rf nlpaug/ rl_cf_proj37/lib/python3.7/site-packages/nlpaug/ 

## You'll need to copy over the model directory for our trained sentiment model, uncompress it into the folder "diego_bert_output_192_v6b/" which test_reinforce.py expects to find in the same folder
    wget http://www.diegoolano.com/files/bert_sentiment_mdl.tar.gz 
    tar xvfz bert_sentiment_mdl.tar.gz

## Finally, to test REINFORCE algorithm on environment run the following 
    python test_reinforce.py output_name num_episodes modeltype

    # where 
    # output_name is the prefix where results will be stored
    # num_episodes is a int
    # modeltype can be "both", "nobase", or "wbase" and determines which models will be run.

## Additionally we have a colab showing some analysis results here:
    https://colab.research.google.com/drive/1xxQcFJ3zwsuiCAqiDDM4tCSciqU3fWJy


## file descriptions
* augmenter.py       -  original work for word substition and comparison that got moved into GYM env

* bert_tflow192.py  -  trains a sentiment classifier based on fine tuning BERT embeddings in TF on the IMDB reviews set and storing up to sequences of 192 length for memory reasons.
                    This gives around 92% accuracy on both the test set and on our counterfactual data set
                    It exports a saved model that is currently in "diego_bert_output_192_v6/export"

* data/             -  folder contains train,dev,test data for CF , along with IMDB data and SST2 review data

* diego_bert_output_192_v6b/export/  - contains trained uncased BERT model for sentiment analysis that we created

* eval_with_tf192.py - loads the saved sentiment model and either evaluates a given labeled set of reviews  or returns predictions/probabilities for unlabeled reviews.
                    We can use this to assess the given state we are in ( ie, the current sentence )

* envs/counterfacs/  - Open AI Gym Custom environment made for Counterfactual Text game

* envs/counterfacs/gym-counterfac/gym_counterfac/envs/counterfac_env.py - Our Custom RL environment code!

* experiments/       - folder for experiments output info

* nlpaug/      -   this is the contains the modified nlpaug package code we use for substitution of words ( files of interest listed)
* nlpaug/augmenter/word/context_word_embs.py   -  see substitute() method for candidate generation

* pos_tagger_flair.py -   POS tagger we use to enforce coherency of changes (ie, only change Verbs with Verbs etc ).  based on https://github.com/zalandoresearch/flair

* requirements.txt     #list of dependencies to install. generated via pip freeze > requirements.txt

* reinforce.py         - This contains our Policy and Value functions for the REINFORCE and REINFORCE with baseline settings

  
* test_reinforce.py   -- This is a wrapper to test reinforce and pass in arguments.


## LEFT FOR FUTURE WORK: to test SAC and TDS implementations run
    git clone https://github.com/openai/spinningup.git
    cd spinningup
    pip install -e .

    python -m spinup.run sac --env counterfac-v0 --exp_name hello_world
