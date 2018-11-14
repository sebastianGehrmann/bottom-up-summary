# Bottom-Up Summarization

This repository describes the process of including Bottom-Up Attention inside your abstractive summarization model. If you are interested in downloading predictions, models or others, please look at the bottom of the page. 

The article will appear in the proceedings of EMNLP 2018. A preprint is available here: [https://arxiv.org/pdf/1808.10792.pdf](https://arxiv.org/pdf/1808.10792.pdf)

If you cite this work, please use the following bibtex:
```
@article{gehrmann2018bottom,
  title={Bottom-Up Abstractive Summarization},
  author={Gehrmann, Sebastian and Deng, Yuntian and Rush, Alexander M},
  journal={arXiv preprint arXiv:1808.10792},
  year={2018}
}
```



## Overview over the whole process

![Image showing the process](https://github.com/sebastianGehrmann/bottom-up-summary/blob/master/bottom-up-summarization.png)


## Individual steps

### (a) Train abstractive model on full data

Please follow the instructions here to train the Pointer-Generator model with Coverage Penalty: [http://opennmt.net/OpenNMT-py/Summarization.html](http://opennmt.net/OpenNMT-py/Summarization.html)

#### Results without Content Selector

CNNDM: R1 39.02, R2 17.25, RL 36.05

Gigaword (Results without penalty): R1 35.51, R2 17.35, RL 33.17

NYT: R1 45.13, R2 30.13, RL 39.67

### (b) Create content-selection dataset

Allennlp requires a specific format of the training data. We provide a script to process a dataset comprising line-separated examples in the form `src.txt` and `tgt.txt`. 

#### Commands

Step 1 - shuffle the data:

``` shuffle_dataset.sh
mkfifo onerandom tworandom
tee onerandom tworandom < /dev/urandom > /dev/null &
shuf --random-source=onerandom ./src.txt > ./src.txt.shuf &
shuf --random-source=tworandom ./tgt.txt > ./tgt.txt.shuf &
wait
```

Step 2 - create data formatted for allennlp

``` preprocess_copy.py
python preprocess_copy.py -src $srcpath
                          -tgt $tgtpath
                          -output data/processed/multicopy.XXX
                          -prune 400 (Max number of words in a document)
                          -num_examples 100000 (100k should be enough for convergence)
```

Preprocessing code can be found in `Extractive Preprocessing.ipynb`.


### (c) Train allenlp tagging model  

#### Commands

Model configuration files are in the folder `allennlp_config`. Modify the lines about file locations and cuda device before running an experiment. 

To train a model, run the command 

```
python -m allennlp.run train 
                       allennlp_config/$filename.json 
                       --serialization-dir $output_folder
```

Make sure to use a different `$output_folder` for each experiment to prevent accidentally overwriting and reusing models. 

There are multiple different configurations in the folder:

- tagger_simple: tagging model with convolutional character encodings and bidirectional LSTM
- tagger_elmo: tagging model with ElMo + standard word encodings and bidirectional LSTM
- tagger_CRF: uses a CRF on top of the model to calculate transitions between states


### (d) Run the Content-Selector

#### Commands

During preprocessing, we create a file named `*.src.txt`. This one can be used to run inference with the trained model. 

```
python -m allennlp.run predict 
                       $modelfile 
                       $datafile 
                       --output $outputfile 
                       --cuda-device 0 
                       --batch-size 50
```


### (e) Use Content-Selector as Extractive Summarizer

One option is to directly use the trained Content-Selector as extractive model. We created a script that takes care of this called `prediction_to_text.py`.

The script can also be used to evaluate against the gold targets as created by the preprocessing by setting `tgt`. You can switch between extraction of sentences and phrases by using the `style` parameter. If you want additional indicators in between extracted phrases, use `divider`. The threshold for the extraction of phrases can be set by `threshold`. Finally, we provide a `prune` option to clip the number of words in an input (you want to use the same number of words as in preprocessing for best results). 


#### Commands

To run, call

```
python prediction_to_text.py -data $predictionfile \
                             -output $outfname \
                             -tgt $tgtfile [optional, prints F1, AUC etc.] \
                             -threshold 0.25 \
                             -divider "" \
                             -style [sentences, phrases, threesent] \
                             -prune 400
```

#### Results

CNNDM with 3 sentences: R1 40.7, R2 18.0, RL 37.0

CNNDM with phrases: R1 42.0, R2 15.9, RL 37.3


### (f) Use probabilities in Bottom-Up Attention

You can find outputs of our model here: https://drive.google.com/file/d/1EqiEVt3H7z7oCQBKkCO7MXoJkXM7Cipr/view?usp=sharing

We are currently working on documenting the code to combine the allennlp output and the OpenNMT model. If you want to run the inference, please download this branch: https://github.com/sebastianGehrmann/OpenNMT-py/tree/copy_constraint 

You will need to run this command: 
```
python translate.py -model $model_PATH                      # You can use a model downloaded from the link above
                    -src $CNNDM_test_input_PATH             # See download link below
                    -constraint_file $allennlp_output_PATH  # Follow instructions or see download link below
                    -threshold $BOTTOM_UP_THRESHOLD         # We found numbers between 0.1 and 0.2 to work. Our reported numbers use 0.15
                    -batch_size 1                           # Currently non-batched, sorry!
                    -min_length 35          
                    -stepwise_penalty 
                    -coverage_penalty summary 
                    -beta 5 
                    -length_penalty wu 
                    -alpha 0.9 
                    -block_ngram_repeat 3 
                    -ignore_when_blocking "." "" "" 
                    -output $prediction_PATH  
                    -gpu $gpuid
```

### Downloadable Content

Note that our predictions have sentence tags `<t>` and `</t>` which need to be removed for ROUGE scoring. To reproduce our numbers, please follow the evaluation instructions [here](http://opennmt.net/OpenNMT-py/Summarization.html).

1) Model: https://s3.amazonaws.com/opennmt-models/Summary/ada6_bridge_oldcopy_tagged_larger_acc_54.84_ppl_10.58_e17.pt
2) allennlp input: https://drive.google.com/file/d/1TNGGBX7iAgvkfyFsDzPKvlbWd4vmCVTR/view?usp=sharing
3) allennlp output: https://drive.google.com/file/d/1IBByzlLwj_JKy-V_mB7563HtRYZilsOl/view?usp=sharing
4) Bottom-Up Attention input: https://drive.google.com/file/d/1k-LqK3Lt7czIKyVrH_tr3P3Qd_39gLhk/view?usp=sharing
5) Bottom-Up Attention output: https://drive.google.com/file/d/1EqiEVt3H7z7oCQBKkCO7MXoJkXM7Cipr/view?usp=sharing
