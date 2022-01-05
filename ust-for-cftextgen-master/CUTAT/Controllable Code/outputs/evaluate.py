from typing import List
from bleu import *
import math
import pandas as pd
import fasttext
import numpy as np
from pathlib import Path
from tqdm import tqdm
from subprocess import PIPE, run
import json

# LM for perplexity
model = None
tokenizer = None


def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

def get_perplexity(text: List[str], dataset: str):
    tmp_corpus = Path("test_corpus_tmp.txt")
    tmp_corpus.write_text("\n".join(text))
    output = out(f"/usr/share/srilm/bin/i686-m64/ngram -lm  "
                f"../srilm/{dataset}.corpus.lm -ppl  test_corpus_tmp.txt")

    ppl = float(output.split("ppl=")[1].split("ppl1")[0])
    ppl1 = float(output.split("ppl1=")[1])

    return ppl

def get_gpt2_perplexity(sentence):
    global model
    if model is None:
        from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
        import torch
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        model.eval()
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss[0].item())


def evaluate(hyp: List[str], ref: List[str], dataset: str):
    eval = {}
    print("Evaluating BLEU...")
    bleu, _ = corpus_bleu(hyp, [[x] for x in ref])
    bleu = bleu[0]
    eval["bleu"] = bleu
    print("\nBLEU: \n", bleu)

    # print("Evaluating PPL (hyp)...")
    # ppls = [get_gpt2_perplexity(s) for s in tqdm(hyp)]
    # ppl = pd.DataFrame(ppls).describe()
    # print("\nPPL: \n", ppl)
    #
    # print("Evaluating PPL (ref)...")
    # ppls_ref = [get_gpt2_perplexity(s) for s in tqdm(ref)]
    # ppl_ref = pd.DataFrame(ppls_ref).describe()
    # print("\nPPL (ref): \n", ppl_ref)

    print("Evaluating PPL (hyp)...")
    ppl = get_perplexity(hyp, dataset)
    eval["ppl"] = ppl
    print("\nPPL: \n", ppl)

    print("Evaluating PPL (ref)...")
    ppl_ref = get_perplexity(ref, dataset)
    eval["ppl_ref"] = ppl_ref
    print("\nPPL: \n", ppl_ref)

    print("Evaluating ACC...")
    labels = ["__label__pos" for _ in range(500)] + ["__label__neg" for _ in range(500)]
    pred_human  = [fasttext_model.predict(l)[0][0] for l in ref]
    pred_model = [fasttext_model.predict(l)[0][0] for l in hyp]
    human_correct = [1 if pred == true else 0 for pred,true in zip(pred_human, labels)] #accuracy of human-made references w.r.t desired labels 
    model_correct = [1 if pred == true else 0 for pred,true in zip(pred_model, labels)] #" model-made changes "
    model_same_as_human = [1 if pred == true else 0 for pred,true in zip(pred_model, pred_human)] #accuracy of model output comapred to class of human reference
    human_acc = np.sum(human_correct) / len(human_correct)
    model_acc = np.sum(model_correct) / len(model_correct)
    model_same_as_human_acc = np.sum(model_same_as_human) / len(model_same_as_human)

    eval["human_acc"] = human_acc
    eval["model_acc"] = model_acc
    eval["model_same_as_human_acc"] = model_same_as_human_acc
    print("\nACC1 (pred on human = labels): \n", human_acc)
    print("\nACC2 (pred on model = labels): \n", model_acc)
    print("\nACC3 (pred on model = pred on human): \n", model_same_as_human_acc)

    return eval


def parse_results(f: Path):
    """
    Load text from results file f and parse.
    File is generated with following format
        gold: ever since joes has changed hands it 's gotten better and better .
        2.0: ever since brother 's gotten worse it has always been just hands worse .
        3.0: ever since brother has gotten worse it 's always been just worse and overpriced .
        4.0: ever since brother has gotten worse it 's always been just hands worse .
        5.0: ever since always has gotten worse it 's always been great and makes overpriced .
        6.0: since always ever works has gotten great prices and it 's always worse .
        7.0: since always has seasoned hands has gotten great prices and it 's always worse .
        8.0: since always has seasoned hands has great prices and it 's always gotten worse .

        gold: there is so much room in that part of the venue
        ...

    """
    examples = [x.split("\n") for x in f.read_text()[1:-1].split("\n\n")]
    model_out = [x[1:] for x in examples]

    results_dict = {x[:3] : [] for x in model_out[0]}
    for example in examples:
        for x in example[1:]:
            results_dict[x[:3]] += [x[5:]]

    results_dict["gold"] = [x[0][6:] for x in examples]
    return results_dict


if __name__ == '__main__':
    DATASET = "yelp"

    original_results = True
    if not original_results:
        # load and parse results
        results_path = Path("../results/yelp_pretrained_output_2019_12_11_15_42_28.txt") #given a raw outputs file, evaluate 
        results_dict = parse_results(results_path) #put the results in the appropriate form for evaulation
        eval_path = results_path.parent / ("eval_" + results_path.name[:-4] + ".json") #output file name
    else:
        hyp_dict = {
            "yelp": "my-model-yelp1.txt",
            "amazon": "my-model-amazon.txt",
            "imagecaption": "my-model-captions.txt"
        }
        ref_dict = {
            "yelp": "human-yelp.txt",
            "amazon": "human-amazon.txt",
            "imagecaption": "human-captions.txt"
        }
        hyp = Path(hyp_dict[DATASET]).read_text().split("\n") #grab the name of the dataset, read the lines
        hyp = [l for l in hyp if l != ""] #strip out all the empty lines in the model outputs
        ref = Path(ref_dict[DATASET]).read_text().split("\n") #do the same thing for the reference gold standards
        ref = [l for l in ref if l != ""]
        results_dict = { #put the results into a form understandable by the metrics - expect format from paper code
            "gold": ref,
            "2.0" : hyp
        }
        eval_path = Path(hyp_dict[DATASET]).parent / ("eval_" + Path(hyp_dict[DATASET]).name + ".json") #output file name

    # classifier for acc
    fasttext_model = fasttext.load_model(f"../fasttext/{DATASET}_model.bin")

    # store eval results in dict
    eval_dict = {}
    ref = results_dict.pop("gold")
    for w, hyp in results_dict.items():
        eval_dict[w] = evaluate(hyp, ref, DATASET)

    eval_path.write_text(json.dumps(eval_dict))
