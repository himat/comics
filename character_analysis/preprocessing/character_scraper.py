import os
import requests
from bs4 import BeautifulSoup
import re
import nltk
import string
from collections import defaultdict

dirname = os.path.dirname(__file__)
csv_heroes_all_file = os.path.join(dirname, "../data/heroes_unfiltered.csv")
csv_heroes_file = os.path.join(dirname, "../data/heroes.csv")
csv_villains_all_file = os.path.join(dirname, "../data/villains_unfiltered.csv")
csv_villains_file = os.path.join(dirname, "../data/villains.csv")

golden_age_protagonists_page = "http://pdsh.wikia.com/wiki/Category:Protagonists"
golden_age_villains_page = "https://en.wikipedia.org/wiki/Category:Golden_Age_supervillains"

hero_excludes = set([])
villain_excludes = set(["Alternative versions of Joker"])

remove_parens_find = r"(.*)\(.*\)"
remove_parens_repl = r"\1"

# Corpus vars
brown_corpus = nltk.corpus.brown
stopwords = nltk.corpus.stopwords.words("english")
min_occurrences = 8 # The top corpus words have at least this many occurrences 

def get_top_corpus_words(corpus, min_occurrences):

    words = corpus.words()
    words = map(lambda w: w.lower(), words)
    words = filter(lambda w: w not in stopwords, words)
    # Replace punctuation with empty string 
    remove_punc_table = dict((ord(char), None) for char in string.punctuation)
    words = [w.translate(remove_punc_table) for w in words]
  
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1
    
    sorted_counts = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)

    num_larger = 0
    for kv in sorted_counts:
        if kv[1] < min_occurrences:
            break
        num_larger += 1

    print(f"{num_larger}/{len(sorted_counts)} ({num_larger/len(sorted_counts)}) words appear at least {min_occurrences} times")

    top_words = set(map(lambda kv: kv[0], sorted_counts[:num_larger]))

    print(f"len: {len(top_words)}")

    return top_words

# Write to CSV in alphabetical order
def write_set(file_name, char_set):
    with open(file_name, "w") as out_file:
        out_file.write("Name\n")
        for char_name in sorted(char_set):
            out_file.write("\"" + char_name + "\"\n")

# process and verify given character name string
# returns false if this character name should not be added to the real results
# also returns the char name with parens and quotes removed, and made lowercase
def process_name(in_char_name, top_corpus_words):

    # Some hero names have the publisher in parens if multiple publishers had comics with the same hero name 
    char_name = re.sub(remove_parens_find, remove_parens_repl, in_char_name).strip() # remove anything in parens
    char_name = char_name.replace('"', '') # remove quotes if any in name
    char_name = char_name.lower()

    if char_name == "question":
        print(char_name, "found")
    if char_name.lower() == "question":
        print(char_name, "found after lower")

    if len(char_name) <= 3: # Ignore short names
        return False, char_name
    if char_name.isdigit(): # Skip if character name is all numbers
        return False, char_name
    if char_name in hero_excludes:
        return False, char_name
    if char_name in villain_excludes:
        return False, char_name
    if char_name in top_corpus_words: # Don't want words that are too common
        return False, char_name

    return True, char_name

def download_protagonists(top_corpus_words):
    response = requests.get(golden_age_protagonists_page)
    root = BeautifulSoup(response.content, "html.parser")

    num_pages = None
    last_page = root.select("div.wikia-paginator li a")[-2]
    last_page_num = int(last_page["data-page"])
    print("Num of pages: ", last_page_num)

    heroes = set()
    heroes_unfiltered = set()

    for page_num in range(1, 1+last_page_num):
        curr_page = golden_age_protagonists_page + "?page=" + str(page_num)
        print("page: ", curr_page)

        response = requests.get(curr_page)
        root = BeautifulSoup(response.content, "html.parser")

        table = root.find("table")
        all_heroes = table.find_all("li")

        num_heroes = len(all_heroes)
        for li in all_heroes:

            add_to_set, hero_name = process_name(li.text, top_corpus_words)
            heroes_unfiltered.add(hero_name)

            if add_to_set:
                heroes.add(hero_name)

        print(f"Page {page_num} has {num_heroes} total heroes")

    print(f"There are {len(heroes_unfiltered)} total heroes (without repeats)")
    print(f"There are {len(heroes)} total heroes (without repeats after filtering)")
    print(f"Excluded heroes: {heroes_unfiltered.difference(heroes)}")

    write_set(csv_heroes_file, heroes)
    write_set(csv_heroes_all_file, heroes_unfiltered)
    print(f"Saved heroes to {csv_heroes_file} and {csv_heroes_all_file}")

def download_villains(top_corpus_words):
    print("\n")

    response = requests.get(golden_age_villains_page)
    print(f"Villains page {golden_age_villains_page}")
    root = BeautifulSoup(response.content, "html.parser")

    names = root.select("div.mw-category-group li")

    villains = set()
    villains_unfiltered = set()

    for name_div in names:

        add_to_set, villain_name = process_name(name_div.text, top_corpus_words)
        villains_unfiltered.add(villain_name)

        if add_to_set:
            villains.add(villain_name) 

    print(f"There are {len(villains_unfiltered)} villains (without repeats)")
    print(f"There are {len(villains)} villains (without repeats after filtering)")
    print(f"Excluded villains: {villains_unfiltered.difference(villains)}")

    write_set(csv_villains_file, villains)
    write_set(csv_villains_all_file, villains)
    print(f"Saved villains to {csv_villains_file} and {csv_villains_all_file}")

if __name__=="__main__":

    top_corpus_words = get_top_corpus_words(brown_corpus, min_occurrences)
    print(len(top_corpus_words))

    if "question" in top_corpus_words:
        print("question in here")

    download_protagonists(top_corpus_words)
    download_villains(top_corpus_words)
