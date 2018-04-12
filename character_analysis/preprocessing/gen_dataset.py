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

dataset_save_file = "../data/character_identity_cloze.csv"

# Dataset variables
TEXT = "text"
NEXT_WORD = "next_word"
IS_CHAR = "is_char"
CHAR_TYPE = "char_type"
CHAR_TYPE_HERO = "hero"
CHAR_TYPE_VILLAIN = "villain"

# Each beginning line in the generated data needs to have at least this many 
#   words so that we don't use lines where the character name appears right at 
#   the beginning of the line, which wouldn't be possible to properly predict
min_num_words_in_line = 3 

# Generates a list of cloze sentences each with a random non-character word removed
def gen_cloze_nonchar_data(comics_text, num_examples, heroes_all_names, villains_all_names):

    assert(len(comics_text) >= num_examples)

    generated_nonchar_data = []

    for line in comics_text:

        if len(generated_nonchar_data) == num_examples:
            break

        words = re.findall(r"\w+", line)
        num_words = len(words)
        

        if num_words <= (min_num_words_in_line+1):
            continue
        

        # print("")
        # randint is inclusive
        chosen_word_ind = random.randint(min_num_words_in_line+1, num_words-1)
        # print(chosen_word_ind)
        chosen_word = words[chosen_word_ind]

        if len(chosen_word) < 2: # word should be at least 2 chars long
            continue

        if chosen_word in heroes_all_names.values or chosen_word in villains_all_names.values:
            continue

        # Get line up to the last occurrence of the word to ensure
        #   line is long enough
        chosen_word_start_pos = line.rfind(chosen_word)
        line_begin = line[:chosen_word_start_pos]
        
        # print(f"{chosen_word} |||| {line}")
        # print("---", line_begin)

        generated_nonchar_data.append({TEXT: line_begin, 
                                       NEXT_WORD: chosen_word,
                                       IS_CHAR: False,
                                       CHAR_TYPE: None})

    if len(generated_nonchar_data) < num_examples:
        raise Exception("not enough examples in data which satisfy requirements")

    return generated_nonchar_data


# Given a sentence and a hero name, returns the sentence up to the hero name 
def gen_cloze_char_sentence(line, name):
    split_pos = line.find(name)
    line_begin = line[:split_pos]

    # print("\nName: ", name)
    # print(line)

    return line_begin
    

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

    print("\n\nGenerating data with character names removed")
    generated_data = []

    for line in comics_text:
        for hero in heroes_names.values:
            words = re.findall(r"\w+", line)
            if hero in words and words.index(hero) > min_num_words_in_line:

                generated_line = gen_cloze_char_sentence(line, hero) 

                generated_data.append({TEXT: generated_line, 
                                       NEXT_WORD: hero,
                                       IS_CHAR: True,
                                       CHAR_TYPE: CHAR_TYPE_HERO})
        for villain in villains_names.values:
            words = re.findall(r"\w+", line)
            if villain in words and words.index(villain) > min_num_words_in_line:

                generated_line = gen_cloze_char_sentence(line, villain) 

                generated_data.append({TEXT: generated_line, 
                                       NEXT_WORD: villain,
                                       IS_CHAR: True,
                                       CHAR_TYPE: CHAR_TYPE_VILLAIN})


        if len(generated_data) == 8000:
            break

    print("\n\nGenerating data with random non-character names removed")
    len_char_data = len(generated_data)
    nonchar_data = gen_cloze_nonchar_data(comics_text, len_char_data, heroes_unfiltered_names, villains_unfiltered_names)
    generated_data += nonchar_data

    print(f"Generated {len_char_data} examples each of positive and negative cloze sentences")

    generated_data_df = pd.DataFrame(generated_data)

    return generated_data_df


if __name__ == "__main__":
    dataset_df = create_dataset()

    dataset_df.to_csv(dataset_save_file, index=False, na_rep="None", columns=[IS_CHAR, CHAR_TYPE, NEXT_WORD, TEXT], quoting=csv.QUOTE_NONNUMERIC)

    print(f"Saved dataset to {dataset_save_file}")



