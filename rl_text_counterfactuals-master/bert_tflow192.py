# Adapted partially from https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
from sklearn.model_selection import train_test_split
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
OUTPUT_DIR = 'diego_bert_output_192_v7'   #v6 uncased 192 gives 91%
DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'polarity'
#MAX_SEQ_LENGTH = 128    
MAX_SEQ_LENGTH = 192    
#MAX_SEQ_LENGTH = 256   # OVERFLOWS MEMORY
#MAX_SEQ_LENGTH = 512   #OVERFLOWS MODEL MEMORY WISE

# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1    # Warmup is a period of time where hte learning rate is small and gradually increases--usually helps training.
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    #dataset = tf.keras.utils.get_file( fname="aclImdb.tar.gz", origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)
    #dataset = "/home/diego/.keras/datasets/"
    dataset = "data/"
    #print(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
    return train_df, test_df

# This is a path to an uncased (all lowercase) version of BERT v6
#BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

# This is a path to a cased version of BERT  v7
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        print(tokenization_info)
        with tf.Session() as sess:
            #vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
          
    return bert.tokenization.FullTokenizer( vocab_file=vocab_file, do_lower_case=do_lower_case)

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    """Creates a classification model."""

    bert_module = hub.Module( BERT_MODEL_HUB, trainable=True)
    bert_inputs = dict( input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module( inputs=bert_inputs, signature="tokens", as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]
    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune data.
    output_weights = tf.get_variable( "output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable( "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

# model_fn_builder actually creates our model function using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):

    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  

        """The `model_fn` for TPUEstimator."""
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model( is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            train_op = bert.optimization.create_optimizer( loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics. 
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score( label_ids, predicted_labels)
                auc = tf.metrics.auc( label_ids, predicted_labels)
                recall = tf.metrics.recall( label_ids, predicted_labels)
                precision = tf.metrics.precision( label_ids, predicted_labels) 
                true_pos = tf.metrics.true_positives( label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives( label_ids, predicted_labels)   
                false_pos = tf.metrics.false_positives( label_ids, predicted_labels)  
                false_neg = tf.metrics.false_negatives( label_ids, predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model( is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            predictions = { 'probabilities': log_probs, 'labels': predicted_labels }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


def load_model():
    # TODO FIGURE OUT HOW TO LOAD SAVED MODEL
    """
    INIT_CHECKPOINT = os.path.join(BERT_FINETUNED_DIR, 'model.ckpt-0')

    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=True,
        use_one_hot_embeddings=True)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE) 
    """
    todo = 1


# Calculate evaluation metrics. 
def eval_metric(label_ids, predicted_labels):
    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
    f1_score = tf.contrib.metrics.f1_score( label_ids, predicted_labels)
    #auc = tf.metrics.auc( label_ids, predicted_labels)
    recall = tf.metrics.recall( label_ids, predicted_labels)
    precision = tf.metrics.precision( label_ids, predicted_labels) 
    true_pos = tf.metrics.true_positives( label_ids, predicted_labels)
    true_neg = tf.metrics.true_negatives( label_ids, predicted_labels)   
    false_pos = tf.metrics.false_positives( label_ids, predicted_labels)  
    false_neg = tf.metrics.false_negatives( label_ids, predicted_labels)
    #"auc": auc,
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

def getPrediction(estimator, tokenizer, in_sentences):
    labels = ["Negative", "Positive"]
    label_list = [0, 1]
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    ret_preds =  [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
    return ret_preds


def get_counterfactual_data():
    #download data
    print("load cf data")
    #cf_dataset = tf.keras.utils.get_file(fname="train_paired.tsv", origin="http://www.diegoolano.com/files/train_paired.tsv", extract=False)
    cf_dataset = "data/all_cf_data.tsv"
    cf_train_df = pd.read_csv(cf_dataset, sep="\t")

    """
    print("tokenize with bert")
    cftoken_text = cf_train_df.apply(lambda x: tokenizer.tokenize(x["Text"]), axis = 1)

    token_lens = [ len( cftoken_text[r] ) for r in range(cf_train_df.shape[0])]
    ta = np.array(token_lens)
    print("Summary stats: max/mean/median/min token lens",np.max(ta), np.mean(ta), np.median(ta), np.min(ta), ta.shape)
    """
    return cf_train_df


# https://stackoverflow.com/questions/56834596/how-to-make-features-for-serving-input-receiver-fn-bert-tensorflow
# https://github.com/google-research/bert/issues/146
def serving_input_fn_old():
    """
      "input_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
      "input_mask" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
      "segment_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
      "label_ids" :  tf.FixedLenFeature([], tf.int64)

      #gives dense error so try with no type
      "input_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int32),
      "input_mask" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int32),
      "segment_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int32),
      "label_ids" :  tf.FixedLenFeature([], tf.int32)

      #with no dtype you get error: TypeError: __new__() missing 1 required positional argument: 'dtype'

      #trying float32.. also gives error
    """

    # https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1   says these should be tf.int32
    """
    feature_spec = {
      "input_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.float32),
      "input_mask" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.float32),
      "segment_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.float32),
      "label_ids" :  tf.FixedLenFeature([], tf.float32)
    }
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_example_tensor')
    receiver_tensors = {'example': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    """

    """
    #again asks for int32
    reciever_tensors = {
        "input_ids": tf.placeholder(dtype=tf.int64, shape=[1, MAX_SEQ_LENGTH])
    }
    features = {
        "input_ids": reciever_tensors['input_ids'],
        "input_mask": 1 - tf.cast(tf.equal(reciever_tensors['input_ids'], 0), dtype=tf.int64),
        "segment_ids": tf.zeros(dtype=tf.int64, shape=[1, MAX_SEQ_LENGTH]),
        "label_ids": tf.zeros(dtype=tf.int64, shape=[])
    }
    return tf.estimator.export.ServingInputReceiver(features, reciever_tensors)
    """
    todo = 1

def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

def main():
    start_time = time.time()
    tf.gfile.MakeDirs(OUTPUT_DIR)

    print("get data which should be pre-cached")
    train, test = download_and_load_datasets()
    label_list = [0, 1]

    #for debugging
    """
    print(type(train),type(test))   #os.path.join(os.path.dirname(dataset), "aclImdb", "train")
    print(dir(train))
    train = train[0:100]
    test = test[0:100]
    """

    print("preprocessing")
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, text_a = x[DATA_COLUMN], text_b = None, label = x[LABEL_COLUMN]), axis = 1)
    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, text_a = x[DATA_COLUMN], text_b = None, label = x[LABEL_COLUMN]), axis = 1)
    
    print("tokenize")
    tokenizer = create_tokenizer_from_hub_module()

    print("get features with BERT")
    # Convert our train and test features to InputFeatures that BERT understands.
    current_time = time.time()
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

    """
    print(type(train_features))  #list each of type <bert.run_classifier.InputFeatures object at 0x7f5107305518>
    print(type(train_features[0]))
    print(train_features[0])
    print(dir(train_features[0]))
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'input_ids', 'input_mask', 'is_real_example', 'label_id', 'segment_ids']

    print(train_features[0].input_ids)
    print(type(train_features[0].input_ids))
    print(type(train_features[0].input_ids[0]))
    print(train_features[0].input_mask)
    print(type(train_features[0].input_mask))
    print(type(train_features[0].input_mask[0]))
    print(train_features[0].label_id)
    print(type(train_features[0].label_id))
    print(train_features[0].segment_ids)
    print(type(train_features[0].segment_ids))

    [101, 2647, 10885, ... , 0,0,0 ]   <class 'list'>   int
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, .., 0, 0, 0] <class 'list'>  int
    0  <class 'int'>
    [0, 0, 0, 0, 0, .. 0,0,0] <class 'list'> int
    """
    
    print("Feature generation took time ", time.time() - current_time)
    # Feature generation took time  211.77728366851807  with MAX_SEQ_LENGHT=128
    # Feature generation took time  213.49941754341125

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    print("create model function and estimator")
    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    #input_ids, input_mask, segment_ids, label_ids,  #from model_fn_builder

    print('Beginning Training!')
    current_time = time.time()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", time.time() - current_time)  
    # Training took time  1354.34 so 22 minutes for 128
    # Training took time  1905.07 so 32 minutes for 192

    print("on test set")
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    print(estimator.evaluate(input_fn=test_input_fn, steps=None))
    # {'eval_accuracy': 0.88828, 'false_positives': 1448.0, 'true_negatives': 11052.0, 'true_positives': 11155.0, 'auc': 0.88828003, 'loss': 0.48115525, 'false_negatives': 1345.0, 'global_step': 2343, 'precision': 0.88510674, 'recall': 0.8924, 'f1_score': 0.8887384}   #FOR 128

    #{'auc': 0.91092, 'loss': 0.38762647, 'f1_score': 0.9112572, 'true_negatives': 11339.0, 'recall': 0.91472, 'eval_accuracy': 0.91092, 'false_positives': 1161.0, 'precision': 0.9078206, 'false_negatives': 1066.0, 'global_step': 2343, 'true_positives': 11434.0}    #FOR 192

    print("NOW TRY ON cf data")
    cf_data = get_counterfactual_data()    #
    pred_sentences = [ cf_data.iloc[a].Text for a in range(cf_data.shape[0]) ] 
    predictions = getPrediction(estimator, tokenizer, pred_sentences)     #[(sentence, prediction['probabilities'], labels[prediction['labels']])]  where labels are "Negative" or "Positive"
    print("Predictions")
    sent_to_int = {"Negative":0, "Positive":1}
    predicted_labels = [ sent_to_int[a[2]] for a in predictions ]     #these should all be 0 and 1 
    #label_ids = [ cf_data.iloc[a].Sentiment for a in range(cf_data.shape[0]) ] 
    label_ids = [ sent_to_int[cf_data.iloc[a].Sentiment] for a in range(cf_data.shape[0]) ] 

    #Predicted_labels 4880 <class 'str'> ['Negative', 'Positive', 'Negative', 'Positive', 'Negative']
    #True_labels 4880 <class 'int'> [0, 1, 0, 1, 0]

    print("Predicted_labels",len(predicted_labels),type(predicted_labels[0]),predicted_labels[0:5])
    print("True_labels",len(label_ids),type(label_ids[0]),label_ids[0:5])
    #failing on AUC curve part so for now just do accuracy

    correct, total = 0, len(label_ids)
    for r in range(len(label_ids)):
        if label_ids[r] == predicted_labels[r]:
            correct += 1
    acc = correct / float(total)

    print("Accuracy", acc) #Accuracy 0.922 for 192 uncased,   Accuracy 0.905327868852459   for 192 cased
    eval_metrics = eval_metric(label_ids, predicted_labels)
    print("Eval metrics: ",eval_metrics)
    print("Elapsed time:", time.time() - start_time)

    #NOW SAVE MODEL
    export_path = '/home/diego/rl_proj_cf/diego_bert_output_192_v7/export'     #v6 is uncased, v7 is cased
    estimator._export_to_tpu = False
    estimator.export_saved_model( export_path, serving_input_receiver_fn=serving_input_fn)

    """
    Elapsed time: 2437.2611441612244
    """
    print("SAVED TO ",export_path)

    #TODO see how it does on SST2 test dataset ( see torch file for URL ).. 


if __name__ == "__main__":
    main()
