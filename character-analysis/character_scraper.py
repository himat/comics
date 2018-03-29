import os
import requests
from bs4 import BeautifulSoup
import re

dirname = os.path.dirname(__file__)
csv_heroes_file = os.path.join(dirname, "heroes.csv")
csv_villains_file = os.path.join(dirname, "villains.csv")

golden_age_protagonists_page = "http://pdsh.wikia.com/wiki/Category:Protagonists"
golden_age_villains_page = "https://en.wikipedia.org/wiki/Category:Golden_Age_supervillains"

hero_excludes = set([])
villain_excludes = set(["Alternative versions of Joker"])

remove_parens_find = r"(.*)\(.*\)"
remove_parens_repl = r"\1"

# process and verify given character name string
# returns false if this character name should not be added to results
def process_name(in_char_name):

    # Some hero names have the publisher in parens if multiple publishers had comics with the same hero name 
    char_name = re.sub(remove_parens_find, remove_parens_repl, in_char_name).strip() # remove anything in parens
    char_name = char_name.replace('"', '') # remove quotes if any in name
    if char_name.isdigit(): # skip if character name is all numbers
        return False, None
    if char_name in hero_excludes:
        return False, None
    if char_name in villain_excludes:
        return False, None

    if len(char_name) <= 2: # Ignore short names
        return False, None

    return True, char_name

def download_protagonists():
    response = requests.get(golden_age_protagonists_page)
    root = BeautifulSoup(response.content, "html.parser")

    num_pages = None
    last_page = root.select("div.wikia-paginator li a")[-2]
    last_page_num = int(last_page["data-page"])
    print("Num of pages: ", last_page_num)

    heroes = set()

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
            if add_to_set:
                heroes.add(hero_name)

        print(f"Page {page_num} has {num_heroes} total heroes")

    print(f"There are {len(heroes)} total heroes (without repeats)")

    # Write to CSV in alphabetical order
    with open(csv_heroes_file, "w") as out_file:
        out_file.write("Name\n")
        for hero in sorted(heroes):
            out_file.write("\"" + hero + "\"\n")

    print(f"Saved heroes to {csv_heroes_file}")

def download_villains():
    response = requests.get(golden_age_villains_page)
    root = BeautifulSoup(response.content, "html.parser")

    names = root.select("div.mw-category-group li")

    villains = set()
    for name_div in names:

        add_to_set, villain_name = process_name(name_div.text)

        if add_to_set:
            villains.add(villain_name) 

    print(f"There are {len(villains)} villains (without repeats)")

    # Write to CSV in alphabetical order
    with open(csv_villains_file, "w") as out_file:
        out_file.write("Name\n")
        for villain in sorted(villains):
            out_file.write("\"" + villain + "\"\n")

    print(f"Saved villains to {csv_villains_file}")

if __name__=="__main__":
    download_protagonists()
    # download_villains()
