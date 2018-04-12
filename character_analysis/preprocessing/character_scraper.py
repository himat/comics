import os
import requests
from bs4 import BeautifulSoup
import re

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

# Write to CSV in alphabetical order
def write_set(file_name, char_set):
    with open(file_name, "w") as out_file:
        out_file.write("Name\n")
        for char_name in sorted(char_set):
            out_file.write("\"" + char_name + "\"\n")

# process and verify given character name string
# returns false if this character name should not be added to the real results
def process_name(in_char_name):

    # Some hero names have the publisher in parens if multiple publishers had comics with the same hero name 
    char_name = re.sub(remove_parens_find, remove_parens_repl, in_char_name).strip() # remove anything in parens
    char_name = char_name.replace('"', '') # remove quotes if any in name

    if char_name.isdigit(): # skip if character name is all numbers
        return False, char_name
    if char_name in hero_excludes:
        return False, char_name
    if char_name in villain_excludes:
        return False, char_name
    if len(char_name) <= 3: # Ignore short names
        return False, char_name

    return True, char_name

def download_protagonists():
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

            add_to_set, hero_name = process_name(li.text)
            heroes_unfiltered.add(hero_name)

            if add_to_set:
                heroes.add(hero_name)

        print(f"Page {page_num} has {num_heroes} total heroes")

    print(f"There are {len(heroes)} total heroes (without repeats)")
    print(f"There are {len(heroes_unfiltered)} total heroes (without repeats and filtering)")

    write_set(csv_heroes_file, heroes)
    write_set(csv_heroes_all_file, heroes_unfiltered)
    print(f"Saved heroes to {csv_heroes_file} and {csv_heroes_all_file}")

def download_villains():
    response = requests.get(golden_age_villains_page)
    print(f"Villains page {golden_age_villains_page}")
    root = BeautifulSoup(response.content, "html.parser")

    names = root.select("div.mw-category-group li")

    villains = set()
    villains_unfiltered = set()

    for name_div in names:

        add_to_set, villain_name = process_name(name_div.text)
        villains_unfiltered.add(villain_name)

        if add_to_set:
            villains.add(villain_name) 

    print(f"There are {len(villains)} villains (without repeats)")
    print(f"There are {len(villains_unfiltered)} villains (without repeats and filtering)")

    write_set(csv_villains_file, villains)
    write_set(csv_villains_all_file, villains)
    print(f"Saved villains to {csv_villains_file} and {csv_villains_all_file}")

if __name__=="__main__":
    download_protagonists()
    download_villains()
