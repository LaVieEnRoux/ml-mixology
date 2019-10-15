"""
Code relating to the creation and serialization of the model needed for
creating new cocktails based on an existing cocktail dataset.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.ensemble

import dataset


def create_all_models(full_dataset, batch_size):
    """
    Create all the models
    """
    ing_feats, ing_labels = full_dataset.get_batch(
        batch_size=batch_size,
        feature_type="ingredient",
    )

    amount_feats, amount_labels = full_dataset.get_batch(
        batch_size=batch_size,
        feature_type="amount",
    )

    ing_model = sklearn.linear_model.LogisticRegression()
    ing_model.fit(ing_feats, ing_labels)

    amount_model = sklearn.linear_model.Ridge(alpha=0.5)
    amount_model.fit(amount_feats, amount_labels)

    return ing_model, amount_model


def add_ingredient(cocktail_vec, ing_model, amount_model):
    """
    Add ingredient to existing cocktail spec
    """
    one_hot_ct = np.copy(cocktail_vec)
    one_hot_ct[np.where(one_hot_ct > 0)] = 1

    # sample according to prediction probabilities
    new_ing_probs = ing_model.predict_proba(np.expand_dims(one_hot_ct, 0)).flatten()
    new_ing_idx = int(np.random.choice(ing_model.classes_, p=new_ing_probs))
    new_ing = np.zeros_like(one_hot_ct)
    new_ing[new_ing_idx] = 1

    amount_feat = np.concatenate((cocktail_vec, new_ing))
    new_ing_amount = amount_model.predict(
        np.expand_dims(amount_feat, 0)
    )
    cocktail_vec[new_ing_idx] = new_ing_amount

    return cocktail_vec


def test_ingredient_model(dataset_dir, batch_size=10000, min_occur=5):
    """
    Test the efficacy of a model on our dataset.
    """
    data, ingredient_set = dataset.load_dataset(dataset_dir, min_occur=min_occur)
    train, valid = dataset.train_valid_split(data)

    print(f"TRAINING SET SHAPE: {train.data.shape}")
    print(f"VALIDATION SET SHAPE: {valid.data.shape}")

    all_train_feats, all_train_labels = train.get_batch(batch_size=batch_size)

    model = sklearn.linear_model.LogisticRegression()
    model.fit(all_train_feats, all_train_labels)

    train_recall_at_1 = recall_at_k(all_train_feats, all_train_labels, model, k=1)
    train_recall_at_3 = recall_at_k(all_train_feats, all_train_labels, model, k=3)
    train_recall_at_5 = recall_at_k(all_train_feats, all_train_labels, model, k=5)
    train_recall_at_20 = recall_at_k(all_train_feats, all_train_labels, model, k=20)

    print(f"TRAIN RECALL AT 1: {train_recall_at_1}")
    print(f"TRAIN RECALL AT 3: {train_recall_at_3}")
    print(f"TRAIN RECALL AT 5: {train_recall_at_5}")
    print(f"TRAIN RECALL AT 20: {train_recall_at_20}")

    all_valid_feats, all_valid_labels = valid.get_batch(batch_size=batch_size)

    valid_recall_at_1 = recall_at_k(all_valid_feats, all_valid_labels, model, k=1)
    valid_recall_at_3 = recall_at_k(all_valid_feats, all_valid_labels, model, k=3)
    valid_recall_at_5 = recall_at_k(all_valid_feats, all_valid_labels, model, k=5)
    valid_recall_at_20 = recall_at_k(all_valid_feats, all_valid_labels, model, k=20)

    print(f"VALID RECALL AT 1: {valid_recall_at_1}")
    print(f"VALID RECALL AT 3: {valid_recall_at_3}")
    print(f"VALID RECALL AT 5: {valid_recall_at_5}")
    print(f"VALID RECALL AT 20: {valid_recall_at_20}")

    return valid_recall_at_5


def test_amount_model(dataset_dir, batch_size=10000, min_occur=5):
    """
    Test the efficacy of a model on our dataset.
    """
    data, ingredient_set = dataset.load_dataset(dataset_dir, min_occur=min_occur)
    train, valid = dataset.train_valid_split(data)

    print(f"TRAINING SET SHAPE: {train.data.shape}")
    print(f"VALIDATION SET SHAPE: {valid.data.shape}")

    all_train_feats, all_train_labels = train.get_batch(
        batch_size=batch_size,
        feature_type="amount",
    )

    model = sklearn.linear_model.Ridge(alpha=0.5)
    model.fit(all_train_feats, all_train_labels)

    train_rmse = rmse(all_train_feats, all_train_labels, model)

    print(f"TRAIN RMSE: {train_rmse}")

    valid_batch_size = 1000
    all_valid_feats, all_valid_labels = valid.get_batch(
        batch_size=valid_batch_size,
        feature_type="amount",
    )

    print(all_valid_feats.shape)
    print(all_valid_labels.shape)

    valid_rmse = rmse(all_valid_feats, all_valid_labels, model)

    print(f"VALID RMSE: {valid_rmse}")

    return valid_rmse


def rmse(features, labels, model):
    """
    Calculate RMSE
    """
    vals = model.predict(features)
    rmse_val = np.sqrt(np.sum((labels - vals) ** 2.0) / labels.size)
    return rmse_val


def recall_at_k(features, labels, model, k=5):
    """
    Calculate recall at k
    """
    probs = model.predict_proba(features)
    top_k_class_indices = np.argsort(probs, axis=1)[:, :-(k + 1):-1]
    preds = model.classes_[top_k_class_indices]

    num_correct = 0.0
    for pred, label in zip(preds, labels):
        if label in pred:
            num_correct += 1.0
    return num_correct / labels.shape[0]


def main():
    """
    Main function
    """
    data, ingredient_set = dataset.load_dataset(
        pathlib.Path("../data/diffords/scratch"), min_occur=18
    )
    full_dataset = dataset.Dataset(data)

    ing_model, amount_model = create_all_models(full_dataset, batch_size=17500)

    while True:
        input_ing = input("Please enter an ingredient (type exit to quit, i for ingredients): ")
        if input_ing.lower() == "exit":
            break
        elif input_ing.lower() == "i":
            for ing_name in ingredient_set:
                print(ing_name)
            continue

        if input_ing not in ingredient_set:
            print("Please select from the ingredient list.")
            continue

        ing_amount = float(input("How much {input_ing} do you want (ml)? "))

        ing_idx = ingredient_set.index(input_ing)
        cocktail_vec = np.zeros((len(ingredient_set)))
        cocktail_vec[ing_idx] = ing_amount

        ing_num = int(input("How many more ingredients will your cocktail have? "))

        for _ in range(ing_num):
            cocktail_vec = add_ingredient(cocktail_vec, ing_model, amount_model)

        print()
        print(dataset.cocktail_to_str(cocktail_vec, ingredient_set))

        print()
        print("Enjoy!")
        print()

if __name__ == "__main__":
    main()
