"""
Defines classes for creating cocktail datasets for the purposes of training
a machine learning model.
"""
import collections
import json
import os
import pathlib

import Levenshtein
import numpy as np
import pandas as pd


class Dataset:
    """
    Defines a class for generating batches of feature+label pairs.

    Ingredient type:
        Given a matrix of cocktail+ingredient vectors, each feature is
        identical to an input vector with ONE ingredient removed. The label
        is a vector containing only the removed ingredient. All are one-hot

    Amount type:
        Given a matrix of cocktail+ingredient vectors, each feature is the
        two vectors: the amount of each ingredient in the cocktail, and the
        one-hot representation of a held out ingredient. The label is the
        float amount of the held out ingredient in the cocktail.
    """
    def __init__(self, data):
        self.data = data

        self.indices = np.arange(self.data.shape[0])
        self.cur_index = 0

    def get_batch(self, batch_size, feature_type="ingredient"):
        """
        Creates a new batch on the fly. Shuffles and rolls over into a new
        epoch if it runs out of data.
        """
        if feature_type == "ingredient":
            feature_shape = (batch_size, self.data.shape[1])
            label_shape = (batch_size)
            feature_func = self._omitted_ing_datapoint
        elif feature_type == "amount":
            feature_shape = (batch_size, self.data.shape[1] * 2)  # concatenated vecs
            label_shape = (batch_size, 1)
            feature_func = self._ingredient_amount_datapoint
        else:
            raise ValueError(f"Invalid feature type: {feature_type}")

        batch_features = np.zeros(feature_shape)
        batch_labels = np.zeros(label_shape)

        for i in range(batch_size):
            # roll over into the next epoch if need be
            if self.cur_index == self.data.shape[0]:
                self.shuffle()

            data_vector, data_label = feature_func()

            batch_features[i, :] = data_vector
            batch_labels[i] = data_label

            self.cur_index += 1

        return batch_features, batch_labels

    def _omitted_ing_datapoint(self):
        """
        Get a data point where the feature vector is an incomplete cocktail
        specification, and the label is an ingredient that completes it.
        """
        data_vector = np.copy(self.data[self.indices[self.cur_index], :])
        ingredient_indices = np.where(data_vector > 0)[0]

        # translate into one-hot here. We might not need this one?
        data_vector[ingredient_indices] = 1

        # sample just a single ingredient from the cocktail to omit
        omitted_ing = np.random.choice(ingredient_indices, size=1)
        data_vector[omitted_ing] = 0

        return data_vector, omitted_ing

    def _ingredient_amount_datapoint(self):
        """
        Get a data point where the feature vector is the ingredient amount for
        each ingredient in the cocktail along with an omitted ingredient, and
        the label is simply the amount of that ingredient.
        """
        data_vector_1 = np.copy(self.data[self.indices[self.cur_index], :])
        ingredient_indices = np.where(data_vector_1 > 0)[0]

        # sample just a single ingredient from the cocktail to predict
        omitted_ing = np.random.choice(ingredient_indices, size=1)
        label = data_vector_1[omitted_ing]
        data_vector_1[omitted_ing] = 0

        data_vector_2 = np.zeros_like(data_vector_1)
        data_vector_2[omitted_ing] = 1

        data_vector = np.concatenate((data_vector_1, data_vector_2))

        return data_vector, label

    def shuffle(self):
        """
        Reshuffle the dataset's indices for pulling batches.
        """
        np.random.shuffle(self.indices)
        self.cur_index = 0


def load_dataset(dataset_dir, min_occur=10):
    """
    Loads the full dataset as a cocktail-ingredient matrix. Returns in either
    one-hot or amount (ml or otherwise) format.
    """
    ingredients = all_ingredients(dataset_dir)

    initial_num_ing = len(ingredients.keys())

    ingredient_set = [k for k, v in ingredients.items() if v > min_occur]
    idx_mapping = {k: v for k, v in zip(ingredient_set, list(range(len(ingredient_set))))}

    final_num_ing = len(ingredient_set)
    print(f"Removed {initial_num_ing - final_num_ing} of {initial_num_ing} ingredients...")

    dataset = []

    all_names = []

    for ct_file in os.listdir(dataset_dir):
        full_ct_path = dataset_dir / ct_file

        with open(full_ct_path, "r") as cocktail_fd:
            cocktail_obj = json.load(cocktail_fd)

        ingredients = cocktail_obj["ingredients"]
        ingredient_names = [x["name"] for x in ingredients]
        ingredient_amounts = [x["amount"] for x in ingredients]

        # if any ingredient is not in ingredient_set, cut out the cocktail
        if not all([x in ingredient_set for x in ingredient_names]):
            continue

        all_names.append(cocktail_obj["name"])

        datapoint = [0] * len(ingredient_set)
        for name, amount in zip(ingredient_names, ingredient_amounts):
            datapoint[idx_mapping[name]] = amount

        dataset.append(datapoint)

    dataset = np.array(dataset)

    ordered_ings = [""] * len(ingredient_set)
    for ing_name, ing_idx in idx_mapping.items():
        ordered_ings[ing_idx] = ing_name

    cocktail_df = pd.DataFrame(
        data=dataset,
        index=all_names,
        columns=ordered_ings,
    )
    cocktail_df.to_csv("./diffords-cocktails.csv")

    return dataset, ingredient_set


def train_valid_split(dataset, split_percentage=0.1):
    """
    Given a dataset matrix, returns the training and testing dataset objects.
    """
    np.random.shuffle(dataset)

    split_idx = int(dataset.shape[0] * split_percentage)
    valid_data = dataset[:split_idx, :]
    train_data = dataset[split_idx:, :]

    return Dataset(train_data), Dataset(valid_data)


def cocktail_to_str(cocktail_vec, ingredient_set):
    """
    Convert a cocktail to a string
    """
    indices = np.argsort(cocktail_vec)[::-1]

    output_str = ""

    for idx in indices:
        if cocktail_vec[idx] == 0.0:
            break
        output_str += f"{cocktail_vec[idx]:.2f} {ingredient_set[idx]}\n"

    return output_str


def all_ingredients(dataset_dir):
    """
    Given a dataset directory, give all the ingredients contained in it in a
    list. Useful for manual inspection + data cleaning.

    Arguments:
        dataset_dir: pathlib.Path
            Path to json cocktail files
    """
    ingredients_dict = collections.defaultdict(int)
    for ct_file in os.listdir(dataset_dir):
        full_ct_path = dataset_dir / ct_file

        with open(full_ct_path, "r") as cocktail_fd:
            cocktail_obj = json.load(cocktail_fd)
        ingredients = cocktail_obj["ingredients"]
        for ingredient in ingredients:
            ingredients_dict[ingredient["name"]] += 1

    # for ingredient in ingredients_dict.keys():
    #     print(ingredient)

    print()
    print("Similar pairs:")
    sim_pairs = set()
    for ingredient in ingredients_dict.keys():
        for ingredient_2 in ingredients_dict.keys():
            if ingredient != ingredient_2 \
               and Levenshtein.ratio(ingredient, ingredient_2) > 0.8:
                sim_pairs.add((ingredient, ingredient_2))

    # for sim_pair in sim_pairs:
    #     print(f"{sim_pair[0]} -------- {sim_pair[1]}")

    return ingredients_dict


def main():
    """
    Main function
    """
    # all_ingredients(pathlib.Path("./diffords/scratch"))

    dataset, ingredient_set = load_dataset(pathlib.Path("../data/diffords/scratch"))
    train, valid = train_valid_split(dataset)
    print(train.get_batch(batch_size=5, feature_type="amount"))

if __name__ == "__main__":
    main()
