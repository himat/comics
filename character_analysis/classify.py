import numpy as np
import pickle

indices_file = "chars_in_text_lines.pkl"


def load_char_indices():
    with open(indices_file, 'rb') as fh:
       character_line_dict = pickle.load(fh)

    return character_line_dict

def classify_single_line():

    character_line_dict = load_char_indices()

    k = ("ace buckley", 'h')

    print(len(character_line_dict[k]))
    print(character_line_dict[k])

    print(len(character_line_dict))

    # for key in character_line_dict:
        # print(key)
        # print(character_line_dict[key])

if __name__ == "__main__":
    classify_single_line()
