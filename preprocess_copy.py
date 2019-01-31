import argparse
import codecs
import json
import numpy as np
import re
# Get a counter for the iterations
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import Counter
print("Loaded libraries...")



parser = argparse.ArgumentParser(
    description="Builds an extractive summary from a json prediction.")

parser.add_argument('-src', required=True, type=str,
                    help="""Path of the src file""")
parser.add_argument('-tgt', required=True, type=str,
                    help="""Path of the tgt file""")

parser.add_argument('-output', type=str,
					default='data/processed/multicopy',
                    help="""Path of the output files""")


parser.add_argument('-prune', type=int, default=200,
                   help="Prune to that number of words.")
parser.add_argument('-num_examples', type=int, default=100000,
                   help="Prune to that number of examples.")

opt = parser.parse_args()

def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end+1])

def format_json(s):
    return json.dumps({'sentence':s})+"\n"

def splits(s, num=200):
    return s.split()[:num]

def make_BIO_tgt(s, t):
    # tsplit = t.split()
    ssplit = s#.split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        if searchstring in t \
            and endix < len(ssplit)-1:
            endix +=1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend(["0"] * (endix-startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix-1, ssplit)
                if matchstrings[full_string] >= 1:
                    matches.extend(["0"]*(endix-startix))
                else:
                    matches.extend(["1"]*(endix-startix))
                    matchstrings[full_string] +=1
                #endix += 1
            startix = endix
    return " ".join(matches)


def main():

	lcounter = 0
	max_total = opt.num_examples

	SOURCE_PATH = opt.src
	TARGET_PATH = opt.tgt

	NEW_TARGET_PATH = opt.output + ".txt"
	PRED_SRC_PATH = opt.output + ".pred.txt"
	PRED_TGT_PATH = opt.output + ".src.txt"

	with codecs.open(SOURCE_PATH, 'r', "utf-8") as sfile:
	    for ix, l in enumerate(sfile):
	        lcounter +=1
	        if lcounter >= max_total:
	            break

	sfile = codecs.open(SOURCE_PATH, 'r', "utf-8")
	tfile = codecs.open(TARGET_PATH, 'r', "utf-8")
	outf = codecs.open(NEW_TARGET_PATH, 'w', "utf-8", buffering=1)
	outf_tgt_src = codecs.open(PRED_SRC_PATH, 'w', "utf-8", buffering=1)
	outf_tgt_tgt = codecs.open(PRED_TGT_PATH, 'w', "utf-8", buffering=1)

	actual_lines = 0
	for ix, (s, t) in tqdm(enumerate(zip(sfile,tfile)), total=lcounter):
	    ssplit = splits(s, num=opt.prune)
	    # Skip empty lines
	    if len(ssplit) < 2 or len(t.split()) < 2:
	        continue
	    else:
	        actual_lines += 1
	    # Build the target
	    tgt = make_BIO_tgt(ssplit,t)
	    # Format for allennlp
	    for token, tag in zip(ssplit, tgt.split()):
	        outf.write(token+"###"+tag + " ")
	    outf.write("\n")
	    # Format for predicting with allennlp
	    outf_tgt_src.write(format_json(" ".join(ssplit)))
	    outf_tgt_tgt.write(tgt + "\n")
	    if actual_lines >= max_total:
	        break

	sfile.close()
	tfile.close()
	outf.close()
	outf_tgt_src.close()
	outf_tgt_tgt.close()



if __name__ == "__main__":
    main()
