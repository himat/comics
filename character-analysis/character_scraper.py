import requests
from bs4 import BeautifulSoup
import re

csv_heroes_file = "heroes.csv"
csv_villains_file = "villains.csv"

golden_age_protagonists_page = "http://pdsh.wikia.com/wiki/Category:Protagonists"
golden_age_villains_page = "https://en.wikipedia.org/wiki/Category:Golden_Age_supervillains"

hero_excludes = set([])
villain_excludes = set(["Alternative versions of Joker"])

remove_parens_find = r"(.*)\(.*\)"
remove_parens_repl = r"\1"

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
            # Some hero names have the publisher in parens if multiple publishers had comics with the same hero name 
            hero = re.sub(remove_parens_find, remove_parens_repl, li.text).strip()
            if hero not in hero_excludes:
                heroes.add(hero)

        print(f"Page {page_num} has {num_heroes} heroes")

    print(f"There are {len(heroes)} total heroes (without repeats)")

    # Write to CSV in alphabetical order
    with open(csv_heroes_file, "w") as out_file:
        for hero in sorted(heroes):
            out_file.write("\"" + hero + "\"\n")

    print(f"Saved heroes to {csv_heroes_file}")

def download_villains():
    response = requests.get(golden_age_villains_page)
    root = BeautifulSoup(response.content, "html.parser")

    names = root.select("div.mw-category-group li")

    villains = set()
    for name_div in names:
        villain = re.sub(remove_parens_find, remove_parens_repl, name_div.text).strip()
        if villain not in villain_excludes:
            villains.add(villain) 

    print(f"There are {len(villains)} villains (without repeats)")

    # Write to CSV in alphabetical order
    with open(csv_villains_file, "w") as out_file:
        for villain in sorted(villains):
            out_file.write("\"" + villain + "\"\n")

    print(f"Saved villains to {csv_heroes_file}")

if __name__=="__main__":
    # download_protagonists()
    download_villains()
