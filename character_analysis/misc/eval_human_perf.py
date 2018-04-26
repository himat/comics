#########
# The purpose of this program is to evaluate a human on a randomly chosen 
#   subset of the test dataset for the character prediction dataset task
#########

import os, sys, argparse
from termcolor import cprint # coloring output

import numpy as np, pandas as pd
import torch

# Needed to import sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import models
from models.character_predictor import CharacterPredictor

dirname = os.path.dirname(__file__)

COL_TEXT = "text"
COL_IS_CHAR = "is_char"
COL_NEXT_WORD = "next_word"
COL_CHAR_TYPE = "char_type"

default_train_file = os.path.join(dirname, "../data/character_identity_cloze_train.csv")
default_test_file = os.path.join(dirname, "../data/character_identity_cloze_test.csv")

def eval_human(num_samples, show_labels):
    
    # Not using default na values since it turns the "nan" string which appears 
    #   in the corpus as a word into an actual NaN 
    train_data_df = pd.read_csv(default_train_file, keep_default_na=False, na_values={COL_NEXT_WORD: ""})
    
    test_data_df = pd.read_csv(default_test_file, keep_default_na=False, na_values={COL_NEXT_WORD: ""})

    test_data_df = test_data_df.reindex(np.random.permutation(test_data_df.index))
    # test_data_df = train_data_df

    chosen_exs = test_data_df[:num_samples]

    user_labels = []

    count = 0

    for ind, ex in chosen_exs.iterrows():
        count += 1
        cprint("\n(" + count + "/" + num_samples + ")  " ex[COL_TEXT], "cyan")

        user_choice = None

        while user_choice is None:
            entered_choice = input("Next word prediction (1 for character, 0 for generic word): ")
            try:
                entered_choice = int(entered_choice)
            except:
                print("Invalid input, use 1 or 0")
                continue

            if entered_choice in [0, 1]:
                user_choice = entered_choice
            else:
                print("Invalid input, use 1 or 0")

        user_labels.append(user_choice)

        if show_labels:
            print(f"True answer: {int(ex[COL_IS_CHAR])}")
            print(f"Next word: {ex[COL_NEXT_WORD]}")


    true_labels = chosen_exs[COL_IS_CHAR].values
    user_labels = np.array(user_labels)

    print("True labels: ", true_labels)
    print("User labels: ", user_labels)

    correct_labels = (true_labels == user_labels)
    num_correct = correct_labels.sum()
    num_total = len(true_labels)

    cprint(f"\nAccuracy: {100 * num_correct / num_total}%", "magenta")


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("--n", type=int, default=200, help="Number of,  examples to have the human be tested on")
    args.add_argument("--show-labels", action="store_true", help="Shows labels to user")

    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()

    eval_human(args.n, args.show_labels)
