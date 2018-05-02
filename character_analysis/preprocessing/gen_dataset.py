#   This file generates a dataset which consists of lines from the comics ocr
# text but which have a word removed. The goal is to then predict whether 
# or not the removed word would have had the name of a superhero/villain or not.
import os, sys, random
import re, csv
import pandas as pd

dirname = os.path.dirname(__file__)
comics_ocr_file = os.path.join(dirname, "../../data/COMICS_ocr_file.csv")
heroes_unfiltered_file = os.path.join(dirname, "../data/heroes_unfiltered.csv")
heroes_file = os.path.join(dirname, "../data/heroes.csv")
villains_unfiltered_file = os.path.join(dirname, "../data/villains_unfiltered.csv")
villains_file = os.path.join(dirname, "../data/villains.csv")

dataset_train_file = "../data/character_identity_cloze_train.csv"
dataset_test_file = "../data/character_identity_cloze_test.csv"

# Dataset variables
COL_TEXT = "text" # X 
COL_IS_CHAR = "is_char" # y
COL_NEXT_WORD = "next_word" 
COL_CHAR_TYPE = "char_type" # hero or villain
COL_POST_TEXT = "post_text" # rest of sentence after the blank, for humans

CHAR_TYPE_HERO = "hero"
CHAR_TYPE_VILLAIN = "villain"

personal_pronouns = ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]
# possessive_pronouns = ["my", "mine", "our", "ours", "its", "his", "her", "hers", "their", "theirs", "your", "yours"]

# How many each of positive and negative examples to generate
num_examples = 19000

# Each beginning line in the generated data needs to have at least this many 
#   words so that we don't use lines where the character name appears right at 
#   the beginning of the line, which wouldn't be possible to properly predict
min_num_words_in_line = 3 

# Negative examples
# Generates a list of cloze sentences each with a random non-character word removed
def gen_cloze_nonchar_data(comics_text, num_examples, heroes_all_names, villains_all_names):
    assert(len(comics_text) >= num_examples)
    print("\n\nGenerating data with random non-character names removed")

    generated_nonchar_data = []

    for line in comics_text:

        if len(generated_nonchar_data) == num_examples:
            break

        words = re.findall(r"\w+", line)
        num_words = len(words)

        if num_words <= (min_num_words_in_line+1):
            continue
        
        # randint is inclusive
        chosen_word_ind = random.randint(min_num_words_in_line+1, num_words-1)
        chosen_word = words[chosen_word_ind]

        if len(chosen_word) < 2: # word should be at least 2 chars long
            continue

        # Don't choose a word that's a hero/villain name
        if chosen_word in heroes_all_names.values or chosen_word in villains_all_names.values:
            continue

        # Don't choose a word that is a personal pronoun, since it is ambiguous
        #   as to whether or not this next word will be a character name or just
        #   a pronoun, so we won't have any negative examples with the next
        #   word being a pronoun
        if chosen_word in personal_pronouns:
            continue

        # Get line up to the last occurrence of the word to ensure
        #   line is long enough
        # TODO: should a regex here be used instead if rfind finds words that 
        #       are a part of other words? It should only find the whole word
        chosen_word_start_pos = line.rfind(chosen_word)
        line_begin = line[:chosen_word_start_pos]
        line_rest = line[chosen_word_start_pos+1:]

        # print(f"{chosen_word} |||| {line}")
        # print("---", line_begin)

        generated_nonchar_data.append({COL_TEXT: line_begin, 
                                       COL_NEXT_WORD: chosen_word,
                                       COL_POST_TEXT: line_rest,
                                       COL_IS_CHAR: False,
                                       COL_CHAR_TYPE: "None"})

    if len(generated_nonchar_data) < num_examples:
        raise Exception("not enough examples in data which satisfy requirements")

    return generated_nonchar_data


# Given a sentence and a hero name, returns the sentence up to the hero name 
def gen_cloze_char_sentence(line, name):
    # Use \b to ensure we match the entire name, and ignore cases where the
    #   name appears inside another word
    re_pattern = r"(.*)\b" + re.escape(name) + r"\b(.*)"     
    re_results = re.search(re_pattern, line)
    line_begin = re_results.group(1).strip()
    line_rest = re_results.group(2).strip()

    # split_pos = line.find(name)
    # line_begin = line[:split_pos]

    return line_begin, line_rest

def gen_cloze_char_data(comics_text, num_examples, heroes_names, villains_names):

    print("\n\nGenerating data with character names removed")
    generated_data_train = []
    generated_data_test = []

    # Use this to alternate which dataset we append to
    # TODO: is this actually a bad idea. Should we put entire comics into either 
    #       just train or test? I think that would make the validation more
    #       representative of the true model performance
    curr_gen_data_ptr = generated_data_train
    def flip_ptr(data_ptr): 
        data_ptr = generated_data_train if data_ptr == generated_data_test else generated_data_test
        return data_ptr

    print_mod = 1000
    prev_print_len = None
    for line in comics_text:
        words = re.findall(r"\w+", line)
        
        for hero in heroes_names.values:
            if hero in words and words.index(hero) > min_num_words_in_line:

                line_begin, line_rest = gen_cloze_char_sentence(line, hero) 

                if line_begin  == "":
                    print("\n-", line_begin, hero, "is char true", "hero")
                    print("words: ", words)
                    split_pos = line.find(hero)
                    o_line_begin = line[:split_pos]
                    print(split_pos)
                    print(o_line_begin)
                    raise ValueError("empty line was generated")

                curr_gen_data_ptr.append({COL_TEXT: line_begin, 
                                          COL_NEXT_WORD: hero,
                                          COL_POST_TEXT: line_rest,
                                          COL_IS_CHAR: True,
                                          COL_CHAR_TYPE: CHAR_TYPE_HERO})
                
                curr_gen_data_ptr = flip_ptr(curr_gen_data_ptr)

        for villain in villains_names.values:
            if villain in words and words.index(villain) > min_num_words_in_line:

                line_begin, line_rest= gen_cloze_char_sentence(line, villain) 

                curr_gen_data_ptr.append({COL_TEXT: line_begin, 
                                          COL_NEXT_WORD: villain,
                                          COL_POST_TEXT: line_rest,
                                          COL_IS_CHAR: True,
                                          COL_CHAR_TYPE: CHAR_TYPE_VILLAIN})

                curr_gen_data_ptr = flip_ptr(curr_gen_data_ptr)


        if len(generated_data_test) == num_examples:
            print(f"Reached data limit of {num_examples}")
            break
        if len(generated_data_test) % print_mod == 0 and len(generated_data_test) != prev_print_len:
            print(f"{len(generated_data_test)}/{num_examples} lines added")
            prev_print_len = len(generated_data_test)

    assert(len(generated_data_train) == len(generated_data_test))

    return generated_data_train, generated_data_test


# Loops through OCR file to build up dataset
def create_dataset():

    # Filter false so it doesn't replace empty strings with NaN
    comics_df = pd.read_csv(comics_ocr_file, na_filter=False)
    comics_text = comics_df.text
    heroes_names = pd.read_csv(heroes_file).Name
    heroes_unfiltered_names = pd.read_csv(heroes_unfiltered_file).Name
    villains_names = pd.read_csv(villains_file).Name
    villains_unfiltered_names = pd.read_csv(villains_unfiltered_file).Name

    heroes_names = heroes_names.str.lower()
    heroes_unfiltered_names = heroes_unfiltered_names.str.lower()
    villains_names = villains_names.str.lower()
    villains_unfiltered_names = villains_unfiltered_names.str.lower()

    generated_data_train, generated_data_test = [], []

    char_data_train, char_data_test = gen_cloze_char_data(comics_text, num_examples, heroes_names, villains_names)
    assert(len(char_data_train) == num_examples and len(char_data_test) == num_examples)
    generated_data_train += char_data_train
    generated_data_test += char_data_test

    # TODO: make gen_cloze_nonchar_data work the same way as for char, where it returns both train/test so you don't have to multiply num_examples by 2
    nonchar_data = gen_cloze_nonchar_data(comics_text, 2*num_examples, heroes_unfiltered_names, villains_unfiltered_names)
    generated_data_train += nonchar_data[:num_examples]
    generated_data_test += nonchar_data[num_examples:]

    print(f"Generated {len(nonchar_data)//2} examples each of positive and negative cloze sentences")
    assert(len(generated_data_train) == len(generated_data_test))

    generated_data_train_df = pd.DataFrame(generated_data_train)
    generated_data_test_df = pd.DataFrame(generated_data_test)

    return generated_data_train_df, generated_data_test_df


if __name__ == "__main__":

    train_dataset_df, test_dataset_df = create_dataset()

    if train_dataset_df.isnull().values.any():
        print(train_dataset_df[train_dataset_df.isnull().any(axis=1)])
        print(train_dataset_df.info())
        raise ValueError("Nulls exist")
    if test_dataset_df.isnull().values.any():
        print(test_dataset_df[test_dataset_df.isnull().any(axis=1)])
        raise ValueError("Nulls exist")

    train_dataset_df.to_csv(dataset_train_file, index=False, na_rep="None", columns=[COL_IS_CHAR, COL_CHAR_TYPE, COL_NEXT_WORD, COL_TEXT, COL_POST_TEXT], quoting=csv.QUOTE_NONNUMERIC)
    test_dataset_df.to_csv(dataset_test_file, index=False, na_rep="None", columns=[COL_IS_CHAR, COL_CHAR_TYPE, COL_NEXT_WORD, COL_TEXT, COL_POST_TEXT], quoting=csv.QUOTE_NONNUMERIC)

    print(f"Saved train dataset to {dataset_train_file}")
    print(f"Saved test dataset to {dataset_test_file}")



