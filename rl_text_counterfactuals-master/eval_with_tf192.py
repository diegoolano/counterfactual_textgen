# https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    from sklearn.model_selection import train_test_split
    import sklearn
    import pandas as pd
    import tensorflow as tf
    tf.test.gpu_device_name()

    import tensorflow_hub as hub
    import time

    import bert
    from bert import run_classifier
    from bert import optimization
    from bert import tokenization

    from tensorflow import keras
    import os
    import re
    import numpy as np

# CONSTANTS
MAX_SEQ_LENGTH = 192    

# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

# I SHOULD HAVE PROBABLY MADE THIS CASED!

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
          
    return bert.tokenization.FullTokenizer( vocab_file=vocab_file, do_lower_case=do_lower_case)

# https://guillaumegenthial.github.io/serving-tensorflow-estimator.html
def load_model():
    # LOAD SAVED MODEL
    label_list = [0, 1]

    from tensorflow.contrib import predictor
    from pathlib import Path

    #export_dir = '/home/diego/rl_proj_cf/diego_bert_output_192_v6/export'
    export_dir = '/home/diego/rl_proj_cf/diego_bert_output_192_v6b/export'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    estimator = predict_fn
    return estimator

def getPrediction(estimator, tokenizer, in_sentences):
    labels = ["Negative", "Positive"]
    label_list = [0, 1]
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    # THIS WORKS FOR A SINGLE EXAMPLE
    #sample_example = {"input_ids":[input_features[0].input_ids], "input_mask": [input_features[0].input_mask], "label_ids":[input_features[0].label_id], "segment_ids":[input_features[0].segment_ids]}

    # FOR MULTIPLE EXAMPLES
    # I NEED TO DO THIS IN BATCHES OTHERWISE I GET AN OUT OF MEMORY ISSUE IF I RUN IT ON ALL 4000

    num_sentences = len(in_sentences)
    if num_sentences < BATCH_SIZE:
        sample_examples = { "input_ids":[input_features[i].input_ids for i in range(num_sentences)], 
                            "input_mask": [input_features[i].input_mask for i in range(num_sentences)], 
                            "label_ids":[input_features[i].label_id for i in range(num_sentences)], 
                            "segment_ids":[input_features[i].segment_ids for i in range(num_sentences)]}
        predictions = estimator(sample_examples) 
        #{'probabilities': array([[-0.56616366, -0.83863366], [-0.8818319 , -0.53447604], [-0.3475206 , -1.2256645 ], [-0.81290066, -0.5862131 ], [-0.6437199 , -0.7451452 ], 
        #                         [-1.4034199 , -0.28203812], [-0.31806448, -1.3003217 ], [-1.0755183 , -0.41721517], [-0.64563936, -0.743025  ], [-1.6141298 , -0.22197396]], dtype=float32), 
        # 'labels': array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int32)}
        #print("Predictions",predictions, len(in_sentences),len(predictions['probabilities']),len(predictions['labels']), num_sentences) 

        #print("Predictions",predictions, len(in_sentences),predictions['probabilities'].shape,predictions['labels'].shape, num_sentences) 
        # Predictions {'labels': 0, 'probabilities': array([[-2.8964018e-03, -5.8457355e+00]], dtype=float32)} 1 (1, 2) () 1
        if num_sentences == 1:
            ret_preds =  [(in_sentences[0], predictions['probabilities'][0], labels[predictions['labels']])]
        else:
            ret_preds =  [(in_sentences[i], predictions['probabilities'][i], labels[predictions['labels'][i]]) for i in range(num_sentences)]
        return ret_preds
    else:
        # DO BATCH PREDs, THEN DO OTHER PARTS
        counter = 0
        final_ret_preds = []
        while (counter + BATCH_SIZE) < num_sentences:
            sample_examples = { "input_ids":[input_features[i].input_ids for i in range(counter,counter + BATCH_SIZE)], 
                                "input_mask": [input_features[i].input_mask for i in range(counter,counter + BATCH_SIZE)], 
                                "label_ids":[input_features[i].label_id for i in range(counter,counter + BATCH_SIZE)], 
                                "segment_ids":[input_features[i].segment_ids for i in range(counter,counter + BATCH_SIZE)]}
            predictions = estimator(sample_examples) 
            ret_preds =  [(in_sentences[i], predictions['probabilities'][i - counter], labels[predictions['labels'][i - counter]]) for i in range(counter,counter + BATCH_SIZE)]
            for rt in ret_preds:
                final_ret_preds.append(rt)
            counter += BATCH_SIZE

        if counter < num_sentences:
            sample_examples = { "input_ids":[input_features[i].input_ids for i in range(counter,num_sentences)], 
                                "input_mask": [input_features[i].input_mask for i in range(counter,num_sentences)], 
                                "label_ids":[input_features[i].label_id for i in range(counter,num_sentences)], 
                                "segment_ids":[input_features[i].segment_ids for i in range(counter,num_sentences)]}
            predictions = estimator(sample_examples) 
            ret_preds =  [(in_sentences[i], predictions['probabilities'][i - counter], labels[predictions['labels'][i - counter]]) for i in range(counter, num_sentences)]
            for rt in ret_preds:
                final_ret_preds.append(rt)

        print("Amount of sentences processed:",len(final_ret_preds),"vs number of input sentences:",num_sentences)
        return final_ret_preds



def get_counterfactual_data():
    print("load cf data")
    cf_dataset = "data/all_cf_data.tsv"
    cf_train_df = pd.read_csv(cf_dataset, sep="\t")
    return cf_train_df

def eval_metric(label_ids, predicted_labels):
    accuracy = sklearn.metrics.accuracy_score(label_ids, predicted_labels)
    f1_score = sklearn.metrics.f1_score( label_ids, predicted_labels)
    recall = sklearn.metrics.recall_score( label_ids, predicted_labels)
    precision = sklearn.metrics.precision_score( label_ids, predicted_labels) 
    confusion_mat = sklearn.metrics.confusion_matrix( label_ids, predicted_labels)
    # in binary classification, the count of true negatives is 0,0 , false negatives is 1,0 , true positives is 1,1  and false positives is 0,1 .
    true_neg = confusion_mat[0,0]
    false_neg = confusion_mat[1,0]
    true_pos = confusion_mat[1,1]
    false_pos = confusion_mat[0,1]
    return {
        "eval_accuracy": accuracy,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "true_positives": true_pos,
        "true_negatives": true_neg,
        "false_positives": false_pos,
        "false_negatives": false_neg
        }

def predict_and_evaluate_sentiment_of_sentences(sentence_pairs=False, debug=False):
    #EXPECTING LIST OF (SENTENCE,SENTIMENT) PAIRS TO PREDICT/EVALUATE
    start_time = time.time()
    if debug:
        print("load tokenizer")

    tokenizer = create_tokenizer_from_hub_module()
    token_time = time.time()

    if debug:
        print("elapsed time",token_time - start_time)

    if debug:
        print("load model")
        #LOAD MODEL AND EVALUATE WITH IT

    estimator = load_model()
    est_time = time.time()

    if debug:
        print("elapsed time",est_time - token_time)

    sent_to_int = {"Negative":0, "Positive":1}
    if sentence_pairs == False:
        if debug:
            print("no sentences passed in so try on cf data")

        cf_data = get_counterfactual_data()    #
        if debug:
            #pred_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ][0:300]  #only keep some for debug
            pred_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ]
        else:
            pred_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ]
        label_ids = [ sent_to_int[cf_data.iloc[a].Sentiment] for a in range(cf_data.shape[0]) ] 
    else:
        if debug:
            print("Running prediction on ",len(sentence_pairs),"sentences")
            print("First of which looks like:",sentence_pairs[0])
        pred_sentences = [a[0] for a in sentence_pairs]
        label_ids = [sent_to_int[a[1]] for a in sentence_pairs]
        
    
    predictions = getPrediction(estimator, tokenizer, pred_sentences)     #[(sentence, prediction['probabilities'], labels[prediction['labels']])]  where labels are "Negative" or "Positive"
    predicted_labels = [ sent_to_int[a[2]] for a in predictions ]     #these should all be 0 and 1 

    if debug:
        print("Predictions")
        print("Predicted_labels",len(predicted_labels),type(predicted_labels[0]),predicted_labels[0:5])
        print("True_labels",len(label_ids),type(label_ids[0]),label_ids[0:5])

    correct, total = 0, len(label_ids)
    falsep, falsen = [],[]
    for r in range(len(predicted_labels)):
        if label_ids[r] == predicted_labels[r]:
            correct += 1
        elif label_ids[r] == 0:
            falsep.append(pred_sentences[r])
        else:
            falsen.append(pred_sentences[r])
            
    acc = correct / float(len(predicted_labels))

    eval_metrics = eval_metric(label_ids[0:len(predicted_labels)], predicted_labels)
    if debug:
        print("Accuracy", acc) #Accuracy 0.922 for 192
        print("Eval metrics: ",eval_metrics)
        
        if len(falsen) > 0:
            toshow = 5
            if len(falsen) < toshow:
                toshow = len(falsen)
            print("Examples of False Negatives (predicted Negative but Sentiment was Positive) found:",falsen[0:toshow])

        if len(falsep) > 0:
            toshow = 5
            if len(falsep) < toshow:
                toshow = len(falsep)
            print("Examples of False Positives (predicted Positive but Sentiment was Negative) found:",falsep[0:toshow])

        print("TOTAL Elapsed time:", time.time() - start_time)

    """ on all 4880 CF EXAMPLES
    Accuracy 0.9145491803278688
    [[2263  177]
     [ 240 2200]]
    Eval metrics:  {'false_positives': 177, 'false_negatives': 240, 'eval_accuracy': 0.9145491803278688, 'precision': 0.9255363904080774, 'true_negatives': 2263, 'f1_score': 0.9134315964293129, 'recall': 0.9016393442622951, 'confmat': array([[2263,  177],
           [ 240, 2200]]), 'true_positives': 2200}
    TOTAL Elapsed time: 63.41726016998291
    """

    """ on 300 first CF EXAMPLES
    Eval metrics:  {'true_positives': 137, 'eval_accuracy': 0.9566666666666667, 'false_negatives': 13, 'false_positives': 0, 'true_negatives': 150, 'recall': 0.9133333333333333, 'precision': 1.0, 'f1_score': 0.9547038327526133}

    Examples of False Negatives (predicted Negative but Sentiment was Positive) found: 
    ['We know from other movies that the actors are good and they make the movie. Not at all a waste of time. The premise was not bad. One workable idea (interaction between real bussiness men and Russian mafia) is followed by an intelligent script', 

    "Well, sorry for the mistake on the one line summary.......Run people, run..to your nearest movie store, that is! This movie is an fabulous!! Imagine! Gary Busey in another low budget movie, with an incredibly funny scenario...isn't that a dream? No (well yes), it is Plato's run...........I give it ****  out of *****.", 

    "The plot was unpredictable, and fighting with guns never gets old, but this is a definate movie to look at if you have a higher IQ and really care about really good movies. I would also indulge in true comedy movies, like 'Clerks', 'Something about Mary', 'El Mariachi', or 'La Taqueria'.", 

    'From the Star of "MITCHELL", From the director of "Joysticks" and "Angel\'s Revenge"!!! These are taglines that would normally make me go see this movie. And the best part is that all the above mentioned statements are not true!!! Ugghhh... Joe Don Baker eats every other five minutes in this film. It\'s like a great remake of "Coogan\'s Bluff"', 

    'The supernatural, vengeful police officer is back for a third installment, this time acting as guardian angel for a wrongfully accused female cop. Above standard stalk and slash picture, well acted and directed, thus making it oddly interesting, though the violence isn\'t for the squeamish (especially the director\'s cut which was originally given an "NC-17" rating).<br /><br />*1/2 out of ****']
    TOTAL Elapsed time: 12.81273865699768

    """

    #TODO how to hide warnings?, soft max probs sent bad and maybe add trust scores as well? )
    #TODO see how it does on SST2 test dataset ( see torch file for URL ).. 

def predict_sentiment_of_sentences(pred_sentences, debug=False):
    #EXPECTING LIST OF SENTENCES TO PREDICT CLASS AND PROBS FOR
    start_time = time.time()
    tokenizer = create_tokenizer_from_hub_module()
    token_time = time.time()
    estimator = load_model()
    est_time = time.time()
    if debug:
        print("Running prediction on ",len(pred_sentences),"sentences")
        print("First of which looks like:",pred_sentences[0])

    predictions = getPrediction(estimator, tokenizer, pred_sentences)     #[(sentence, prediction['probabilities'], labels[prediction['labels']])]  where labels are "Negative" or "Positive"
    predicted_labels = [ a[2] for a in predictions ]     
    predicted_probs = [ a[1] for a in predictions ]     

    if debug:
        toshow = 5
        if len(predicted_labels) < toshow:
            toshow = len(predicted_labels)
        print("Some Sentences with predicted_labels and probs")
        for i in range(toshow):
            print(i,pred_sentences[i],"PREDICTED:",predicted_labels[i],predicted_probs[i])
    return {"sentences": pred_sentences, "predicted_labels": predicted_labels, "predicted_probs": predicted_probs}

if __name__ == "__main__":
    #1. SHOW PRED ON CF DATA
    predict_and_evaluate_sentiment_of_sentences(False,True)

    #2. SHOW PREDS ON LABELED DATA
    #test_sentences = [("I really think the film Home Alone sucks.","Negative"), ("I absolutely loved the plot of the Lion King","Positive")]
    #predict_and_evaluate_sentiment_of_sentences(test_sentences,debug=True)

    #3. SHOW PREDS/PROBS ON UNLABELLED DATA
    #more_test_sentences = ["I really think the film Home Alone is lacking in plot and acting, but i still found the overall effort worth it.", "I loved the plot of the Lion King, but hate animation films in general and left without my opinion unchanged", "The Terminator is the opposite of what I think a good film should be.", "I loved the Mighty Ducks", "I hated the Irishman." ]
    #results = predict_sentiment_of_sentences(more_test_sentences, True)
    #print(results)
    todo = 1

    """
    Some Sentences with predicted_labels and probs
    0 I really think the film Home Alone is lacking in plot and acting, but i still found the overall effort worth it. PREDICTED: Positive [-6.5107002e+00 -1.4885309e-03]
    1 I loved the plot of the Lion King, but hate animation films in general and left without my opinion unchanged PREDICTED: Negative [-0.02475958 -3.7108953 ]
    2 The Terminator is the opposite of what I think a good film should be. PREDICTED: Negative [-0.02276443 -3.7939146 ]
    3 I loved the Mighty Ducks PREDICTED: Positive [-3.851096   -0.02148555]
    4 I hated the Irishman. PREDICTED: Negative [-0.06838427 -2.71661   ]
    """
