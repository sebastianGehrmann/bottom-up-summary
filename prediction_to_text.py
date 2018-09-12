import argparse
import codecs
import json
import numpy as np
import re
# Get a counter for the iterations
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import Counter
# For tgt
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
print("Loaded libraries...")


parser = argparse.ArgumentParser(
    description="Builds an extractive summary from a json prediction.")

parser.add_argument('-data', required=True,
                    help="""Path of the json file""")
parser.add_argument('-output', type=str, default="",
                    help="""Path of the output file""")
parser.add_argument('-tgt', type=str, default="",
                    help="Also prints statistics (F1/AUC)")
parser.add_argument('-threshold', type=float, default=.25,
                    help="Threshold for extracting a word.")
parser.add_argument('-divider', type=str, default="",
                    help="Divider between phrases.")
parser.add_argument('-style', default='phrases',
                    choices=['phrases', 'sentences', 'threesent'],
                    help="""Which style of processing.""")
parser.add_argument('-prune', type=int, default=200,
                   help="Prune to that number of words.")


opt = parser.parse_args()


def get_sents(words, tags, probs):
    # 1: divide into sentences with associated probabilities
    STARTTOKEN = "<t>"
    ENDTOKEN = "</t>"

    full_text = ""
    so_far = []
    current_tags = []
    current_probs = []

    highest_so_far = []
    avg_of_highest = 0.
    for word, tag, prob in zip(words, tags, probs):
        so_far.append(word)
        current_tags.append(tag)
        current_probs.append(prob)
        if word == ".":
            if sum(current_tags) > 1:
                full_text += STARTTOKEN + " " \
                             + " ".join(so_far)\
                             + " " + ENDTOKEN + " "
            if np.max(current_probs) > avg_of_highest:
                highest_so_far = so_far.copy()
                avg_of_highest = np.mean(current_probs)

            current_tags = []
            so_far = []
            current_probs = []
    if full_text == "":
        full_text =  STARTTOKEN + " " \
                     + " ".join(highest_so_far)\
                     + " " + ENDTOKEN + " "
    return full_text

def get_phrases(words, tags):
    prev = 0
    pred = []
    for word, tag in zip(words, tags):
        if tag == 1:
            pred.append(word)
        elif prev == 1 and opt.divider != "":
            pred.append(opt.divider)
        prev = tag
    pred = " ".join(pred)
    return pred

def get_three(words, outf, probs):
    # 1: divide into sentences with associated probabilities
    STARTTOKEN = "<t>"
    ENDTOKEN = "</t>"

    sents = []
    scores = []
    current_sent = []
    current_probs = []
    current_num = 0

    # First get all sentences and associated avg copy scores
    for word, prob in zip(words, probs):
        current_sent.append(word)
        current_probs.append(prob)
        if word == "." or word == "!" or word == "?":
            scores.append((np.mean(current_probs), current_num))
            sent = STARTTOKEN + " " \
                     + " ".join(current_sent)\
                     + " " + ENDTOKEN
            sents.append(sent)

            current_sent=[]
            current_num+=1

    # Now select 3 top ones
    scores.sort(key=lambda x:x[0])
    # print (scores)
    top3 = scores[-3:]
    top3.sort(key=lambda x: x[1])

    for i in top3:
        outf.write(str(i[1]) + "\n")
    # print(top3)
    # print([i[1] for i in top3])
    full_text = " ".join([sents[i[1]] for i in top3])
    # print(full_text)
    return full_text

def read_tgt_file(fname):
    tgt = []
    with codecs.open(fname, "r") as f:
        for l in f:
            tgt.append([int(i) for i in l.split()][:opt.prune])
    return tgt


def main():

    # Get a line counter
    lcounter = 0
    with codecs.open(opt.data, 'r', "utf-8") as sfile:
        for ix, l in enumerate(sfile):
            lcounter +=1

    resfile = codecs.open(opt.data, 'r')
    if opt.output:
        outfile = codecs.open(opt.output, 'w')
        outfile2 = codecs.open(opt.output + ".track", "w")

    if opt.tgt:
        y = read_tgt_file(opt.tgt)
        yhat = []
        yprobs = []

    for ix, line in tqdm(enumerate(resfile), total=lcounter):
        cline = json.loads(line)
        words = cline['words']
        # print("len words", len(words))
        probs = [p[1] for p in cline['class_probabilities'][:len(words)]]
        tags = [1 if p > opt.threshold else 0 for p in probs]

        if opt.output:
            if opt.style == "phrases":
                pred = get_phrases(words, tags)
            elif opt.style == "sentences":
                pred = get_sents(words, tags, probs)
            elif opt.style == "threesent":
                    pred = get_three(words, outfile2, probs)
            outfile.write(pred + "\n")

        if opt.tgt:
            yhat.append(tags)
            yprobs.append(probs)

        # if ix > 150:
        #     break
    if opt.tgt:
        print("Evaluating Model...")
        y_flat = []
        yhat_flat = []
        probs_flat = []
        for tgt, pred, pr  in zip(y, yhat, yprobs):
            if len(pred) != len(tgt):
                pass
                # print("woa", len(pred), len(tgt))
            else:
                for t, p, cpr in zip(tgt, pred, pr):
                    y_flat.append(t)
                    yhat_flat.append(p)
                    probs_flat.append(cpr)
        y_flat = np.array(y_flat)
        yhat_flat = np.array(yhat_flat)
        probs_flat = np.array(probs_flat)
        fpr, tpr, thresholds = roc_curve(y_flat, probs_flat)
        print("AUC: {:.1f}".format(auc(fpr, tpr)*100))
        print("F1-Score (binary): {:.2f}".format(f1_score(y_flat, yhat_flat)*100))
        print("Confusion Matrix:")
        print(confusion_matrix(y_flat, yhat_flat))

        tgt_names = ["O", "I"]
        print("Classification Report:")
        print(classification_report(y_flat, yhat_flat, target_names=tgt_names))


    resfile.close()
    if opt.output:
        outfile.close()
        outfile2.close()

if __name__ == "__main__":
    main()

