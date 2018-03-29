import os, sys
import pandas as pd 
from collections import defaultdict
import pickle

dirname = os.path.dirname(__file__)
comics_ocr_file = os.path.join(dirname, "../data/COMICS_ocr_file.csv")
heroes_file = os.path.join(dirname, "./heroes.csv")
villains_file = os.path.join(dirname, "./villains.csv")
save_indices_file = "chars_in_text_lines.pkl"

# Finds each character in the text and prints them with the number of lines they were found in 
def print_all_char_counts():
   
    # Filter false so it doesn't replace empty strings with NaN
    comics_df = pd.read_csv(comics_ocr_file, na_filter=False)
    comics_text = comics_df.text
    heroes_names = pd.read_csv(heroes_file).Name
    villains_names = pd.read_csv(villains_file).Name

    heroes_names = heroes_names.str.lower()
    villains_names = villains_names.str.lower()

    heroes_string = ""
    for hero_name in heroes_names.values:
        heroes_string += hero_name + " " 

    # print(comics_text.dtypes)
    # print(comics_text.head)
    # print(heroes_names.head)
    # print(heroes_string[:100])
    # print(villains_names.head)

    # name = "zudo"
    name = "wilson"
    # found = True if name in heroes_names.values else False
    # print(comics_text[9])
    found = name in comics_text[9]

    # print("name found?: ", found)

    found_heroes = defaultdict(int)
    found_villains = defaultdict(int)

    for hero in heroes_names.values:
        hero = hero.lower()

        for line in comics_text:
            if hero in line:
                found_heroes[hero] += 1

    for villain in villains_names.values:
        villain = villain.lower()

        for line in comics_text:
            if villain in line:
                found_villains[villain] += 1

    sorted_hero_counts = sorted(found_heroes.items(), key=lambda x: x[1], reverse=True)
    sorted_villain_counts = sorted(found_villains.items(), key=lambda x: x[1], reverse=True)
    print(f"\nNumber of given heroes found in comics text: {len(found_heroes)} / {len(heroes_names)}")
    print("Found heroes: ", sorted_hero_counts)
    print(f"\nNumber of given villains found in comics text: {len(found_villains)} / {len(villains_names)}")
    print("Found villains: ", sorted_villain_counts)

# Finds the lines associated with a character and saves them to a file
def save_char_indices():

    # Filter false so it doesn't replace empty strings with NaN
    comics_df = pd.read_csv(comics_ocr_file, na_filter=False)
    comics_text = comics_df.text
    heroes_names = pd.read_csv(heroes_file).Name
    villains_names = pd.read_csv(villains_file).Name

    heroes_names = heroes_names.str.lower()
    villains_names = villains_names.str.lower()

    found_heroes = defaultdict(list)
    found_villains = defaultdict(list)

    # limit = 5
    # done = False

    for hero in heroes_names.values:

        for (i, line) in enumerate(comics_text):
            if hero in line:
                found_heroes[(hero,'h')].append(line)

            # if limit < 0:
                # done = True
                # break

        # limit -= 1
        # if done:
            # break

    # for villain in villains_names.values:

        # for line in comics_text:
            # if villain in line:
                # found_villains[villain] += 1

    # Save results
    with open(save_indices_file, 'wb') as fh:
        pickle.dump(found_heroes, fh)

if __name__ == "__main__":
    # print_all_char_counts()
    save_char_indices()
