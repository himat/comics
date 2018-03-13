import requests
from bs4 import BeautifulSoup
import re

csv_heroes_file = "heroes.csv"

golden_age_protagonists_page = "http://pdsh.wikia.com/wiki/Category:Protagonists"

response = requests.get(golden_age_protagonists_page)
root = BeautifulSoup(response.content, "html.parser")


num_pages = None
last_page = root.select("div.wikia-paginator li a")[-2]
last_page_num = int(last_page["data-page"])
print("Num of pages: ", last_page_num)

remove_quotes_find = r"(.*)\(.*\)"
remove_quotes_repl = r"\1"

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
        hero = re.sub(remove_quotes_find, remove_quotes_repl, li.text).strip()
        if hero == "13":
            print("13 on ", page_num)
        heroes.add(hero)

    print(f"Page {page_num} has {num_heroes} heroes")

print(f"There are {len(heroes)} total heroes (without repeats)")

# Write to CSV in alphabetical order
with open(csv_heroes_file, "w") as out_file:
    for hero in sorted(heroes):
        out_file.write(hero + "\n")

print(f"Saved heroes to {csv_heroes_file}")
