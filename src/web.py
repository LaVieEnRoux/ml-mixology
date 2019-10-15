"""
Defines utilities for loading in datasets from the web.
"""
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

import json
import os
import pathlib
import time

import bs4
import requests

WAIT_TIME_SEC = 0.75

with open("config.json", "r") as cfg:
    DATASET_CONFIG = json.load(cfg)

GLASS_TO_TOP_UP = DATASET_CONFIG["top_up_vals"]
INVALID_NUMS = DATASET_CONFIG["invalid_page_nums"]
INGREDIENT_MAPPINGS = DATASET_CONFIG["ingredient_mappings"]
MLS_IN_SPOON = DATASET_CONFIG["mls_in_barspoon"]
MLS_IN_DASH = DATASET_CONFIG["mls_in_dash"]
MLS_IN_FLOAT = DATASET_CONFIG["mls_in_float"]
MLS_IN_SPLASH = DATASET_CONFIG["mls_in_splash"]
MLS_IN_CUPFUL = DATASET_CONFIG["mls_in_cupful"]


def pull_diffords(data_dir, html_dir):
    """
    Pulls data from the Difford's Cocktail Guide website into a local data
    format

    Works with the Difford's website as of April 21st, 2019

    Arguments:
        data_dir: str
            The caching location for storing the cocktail files
        html_dir: str
            The caching location for storing the cocktail page HTML response
    """
    page_num = 0
    while True:
        page_num += 1

        # some of these pages always 404
        if page_num in INVALID_NUMS:
            continue

        # skip anything we already have the cocktail JSON for
        recipe_location = pathlib.Path(data_dir) / f"{page_num}.json"
        if os.path.isfile(recipe_location):
            continue

        # don't hit the URL if it's already been saved
        cache_location = pathlib.Path(html_dir) / f"{page_num}.html"
        if os.path.isfile(cache_location):
            print(f"Loading {page_num} from local HTML cache...")
            with open(cache_location, "r") as html_file:
                html_text = html_file.read()
        else:
            # don't wanna hit the website too often
            time.sleep(WAIT_TIME_SEC)
            try:
                print(f"Loading {page_num}...")
                html_text = requests.get(
                    f"https://www.diffordsguide.com/cocktails/recipe/{page_num}/foobar"
                ).text
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Request failed with exception {exc}")
                continue

            with open(cache_location, "w") as html_file:
                html_file.write(html_text)

        soup = bs4.BeautifulSoup(html_text, "html.parser")
        name = soup.find("h1", itemprop="name")
        if name is None:
            # this number doesn't have a cocktail
            print(f"Cocktail num {page_num} doesn't work.")
            with open("config.json", "w") as config_fp:
                DATASET_CONFIG["invalid_page_nums"].append(page_num)
                json.dump(DATASET_CONFIG, config_fp, indent=4)
            continue
        name = name.text

        ingredients = soup.find("table", id="cocktails_recipe_ingredients_table")
        ingredient_list = []
        glass = ""
        for table_row in ingredients.find_all("tr"):
            # is this a row for the glass, or for the ingredient?
            if table_row.find("td", class_="cocktails_recipe_glass") is not None:
                glass = table_row.find("a").text.rsplit(" ", 1)[0].lower()
            else:
                # the ingredient summaries are inconsistent -- collapse them here
                summary = table_row.find("meta").get("content", "").strip().lower()
                summary = summary.replace("(optional)", "").strip()

                if "top up with" in table_row.text.lower():
                    # infer top up amount based on glass type
                    try:
                        summary = GLASS_TO_TOP_UP[glass] + " " + summary
                    except KeyError:
                        print(f"WE DON'T HAVE A TOP UP RULE FOR GLASS: {glass}")
                        continue

                if "spoon" in table_row.text.lower():
                    # ingredient is listed in spoons but not in ml!
                    num_spoons, summary = summary.split(" ", 1)
                    ml_amount = amount_to_num(num_spoons) * MLS_IN_SPOON
                    summary = f"{ml_amount} ml {summary}"

                if "dash" in table_row.text.lower():
                    # ingredient is listed in dashes but not in ml!
                    num_dashes, summary = summary.split(" ", 1)
                    ml_amount = amount_to_num(num_dashes) * MLS_IN_DASH
                    summary = f"{ml_amount} ml {summary}"

                if "float" in table_row.text.lower():
                    # ingredient is floated on top, no amount
                    summary = f"{MLS_IN_FLOAT} ml {summary}"

                if "splash" in table_row.text.lower():
                    # ingredient is splashed on top, no amount
                    summary = f"{MLS_IN_SPLASH} ml {summary}"

                if "cupful" in table_row.text.lower():
                    # lol damn
                    summary = f"{MLS_IN_CUPFUL} ml {summary}"

                ingredient_list.append(summary)

        instructions = soup.find("p", itemprop="recipeInstructions")
        if instructions is None:
            instructions = soup.find("p", itemprop="recipeInstructions description")
        instructions = instructions.text.lower().strip()

        # save to json via dict
        cocktail = {}
        cocktail["name"] = name
        cocktail["ingredients"] = [
            ingredient_to_dict(ing) for ing in ingredient_list
        ]
        cocktail["glass"] = glass
        cocktail["instructions"] = instructions

        recipe_location = pathlib.Path(data_dir) / f"{page_num}.json"
        with open(recipe_location, "w") as recipe_file:
            json.dump(cocktail, recipe_file, indent=4)


def ingredient_to_dict(ingredient):
    """
    Convert an ingredient string from Difford's into an ingredient obj
    """
    print(ingredient)
    ingredient_fields = ingredient.split(" ", 1)
    if len(ingredient_fields) == 1:
        # some ingredients are listed as just one word
        amount = "1"
        name = ingredient_fields[0]
    else:
        amount = ingredient_fields[0]
        name = ingredient_fields[1]

    # sometimes these ingredient amounts are written with the fraction chars...
    amount = amount_to_num(amount)

    # some ingredients are mislabelled or are BASICALLY the same thing
    alternate_name = INGREDIENT_MAPPINGS.get(name, None)
    if alternate_name:
        name = alternate_name

    return {"amount": amount, "name": name}


def amount_to_num(amount):
    """
    Amounts use odd characters sometimes.
    """
    amount_dict = {
        "⅛": 0.125,
        "¼": 0.25,
        "½": 0.5,
        "¾": 0.75,
        "1½": 1.5,
        "⅔": 0.67,
        "1¼": 1.25,
        "2½": 2.5,
        "⅚": 0.84,
        "⅓": 0.33,
        "1/12": 1.0 / 12.0,
    }
    if amount in amount_dict.keys():
        return amount_dict[amount]

    return float(amount)


def main():
    """
    Main function.
    """
    pull_diffords(data_dir="../data/diffords/scratch", html_dir="../data/diffords/html")


if __name__ == "__main__":
    main()
