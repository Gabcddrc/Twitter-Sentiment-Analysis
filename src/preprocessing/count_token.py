## Count token with their number of occurances in positive and negative
## tweets to help identify non-discriminatory stop words specific for the given
## data.

import os

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import pandas as pd
from operator import itemgetter

# Custom function to load pre-split train, validation and test data as pandas
# DataFrames.
import sys

sys.path.append("../")
from data_processing.loading import load_train_valid_data


def main():
    experiment = Experiment(
        api_key="",
        project_name="cil-project",
        workspace="smueksch",
        # Prevent CometML from dumping all sorts of parameter info about SVC,
        # embedding, etc. into the experiment log. Avoids cluttering but could
        # be preferrable to set to True.
        auto_param_logging=False,
        disabled=True,
    )
    experiment.set_name("Count Token (Train + Valid)")

    # Need to provide the relative path to the folder containing the
    # training_set.csv and validation_set.csv
    path_to_dataset = os.path.join(os.pardir, os.pardir, "dataset")
    train, valid = load_train_valid_data(path_to_dataset)
    train_valid = pd.concat([train, valid])

    # MAP tweet data into occurances of the form [token, label, 1], where 1 is supposed to
    # represent a count that can later be added up.

    token_occurances = []

    def enumerate_occurances(tweet: str, label: int) -> None:
        for token in tweet.split():
            token_occurances.append([token, label, 1])

    train_valid.apply(
        lambda row: enumerate_occurances(row["tweet"], row["label"]), axis=1
    )

    # SORT the occurances by token so that they can be reduce to total_counts.

    sorted_token_occurances = sorted(token_occurances, key=itemgetter(0))

    # REDUCE occurance count by token to a form of [token, positive_count, negative_count].

    collected_occurances = []
    current_token = sorted_token_occurances[0][0]
    num_pos_label = 0
    num_neg_label = 0

    for occurance in sorted_token_occurances:
        token, label, count = occurance
        if current_token != token:
            # New token, write data and reset.
            collected_occurances.append([current_token, num_pos_label, num_neg_label])
            current_token = token
            num_pos_label = count if label == 1 else 0
            num_neg_label = count if label == -1 else 0
        else:
            # Same token, count how often it is positive or negative.
            num_pos_label += count if label == 1 else 0
            num_neg_label += count if label == -1 else 0

    # Flush the last collected data.
    collected_occurances.append([current_token, num_pos_label, num_neg_label])

    token_count = pd.DataFrame(
        collected_occurances, columns=["token", "pos_count", "neg_count"]
    )

    experiment.log_table(filename="token_count.csv", tabular_data=token_count)
    token_count.to_csv("token_count.csv")


if "__main__" == __name__:
    main()
