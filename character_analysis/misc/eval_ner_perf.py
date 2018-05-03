#############################################
# This file uses an existing NER system to try to identify character names
#   in a given sentence from the character identity cloze dataset
#############################################

import os, sys, argparse
import numpy as np, pandas as pd

from termcolor import cprint # coloring output

from nltk.tag.stanford import StanfordNERTagger
from nltk import tokenize 
import nltk

dirname = os.path.dirname(__file__)

COL_TEXT = "text" # x
COL_IS_CHAR = "is_char" # y
COL_NEXT_WORD = "next_word"
COL_CHAR_TYPE = "char_type"
COL_POST_TEXT = "post_text"


default_train_file = os.path.join(dirname, "../data/character_identity_cloze_train.csv")
default_test_file = os.path.join(dirname, "../data/character_identity_cloze_test.csv")
stanford_classifier_file = "../data/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz"
stanford_jar_file = "../data/stanford-ner/stanford-ner.jar"

tagger = StanfordNERTagger(stanford_classifier_file, stanford_jar_file, encoding="utf-8")

def eval_ner(args):

    # Not using default na values since it turns the "nan" string which appears 
    #   in the corpus as a word into an actual NaN 
    # train_data_df = pd.read_csv(default_train_file, keep_default_na=False, na_values={COL_NEXT_WORD: ""})
    
    test_data_df = pd.read_csv(default_test_file, keep_default_na=False, na_values={COL_NEXT_WORD: ""})

    test_data_df = test_data_df.reindex(np.random.permutation(test_data_df.index))

    all_entity_tags = ["LOCATION", "PERSON", "ORGANIZATION"]

    count_print_mod = max(len(test_data_df) / 100, 1)
    count = 0
    predicted_labels = []
    for ind, ex in test_data_df.iterrows():

        next_word = ex[COL_NEXT_WORD]
    
        # TODO: see if performance changes if you input the entire sentence up to and including the next word vs just the next word by itself
        des_text = next_word
        

        tok_text = [tokenize.word_tokenize(des_text)]

        classified_text = tagger.tag_sents(tok_text)

        classified_tag = classified_text[0][0][1]
        assert(classified_tag in (["O"] + all_entity_tags))

        if classified_tag in all_entity_tags:
            pred_label = 1
        else:
            pred_label = 0

        if count % count_print_mod == 0:
            print(f"{count}/{len(test_data_df)}")
        if args.print:
            cprint(f"\nNext word: {des_text}", "cyan")
            print("classified text: ", classified_text)
            print("Pred label: ", pred_label)
            print("True label: ", ex[COL_IS_CHAR])

        predicted_labels.append(pred_label)

        count += 1
 
    # Evaluate 
    true_labels_arr = test_data_df[COL_IS_CHAR].values
    pred_labels_arr = np.array(predicted_labels)

    assert(len(true_labels_arr) == test_data_df.shape[0])
    assert(len(true_labels_arr) == len(pred_labels_arr))
    
    num_correct = (true_labels_arr == pred_labels_arr).sum()
    num_total = len(test_data_df)

    print("\n\n\n")
    print(f"Accuracy of NER tagger: {num_correct}/{num_total} ({100 * num_correct/num_total}%)")

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("--print", action="store_true", help="Shows text predictions and true labels")

    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    eval_ner(args)
