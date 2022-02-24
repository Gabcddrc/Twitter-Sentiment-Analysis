## Count token with their number of occurances in positive and negative
## tweets to help identify non-discriminatory stop words specific for the given
## data.

import os

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import re
import yaml

# Custom function to load pre-split train, validation and test data as pandas
# DataFrames.
import sys

sys.path.append("../")
from data_processing.loading import load_train_valid_data, load_test_data


class Normalizer:
    """Base class for normalization with metrics."""

    def __init__(self) -> None:
        self.metric = 0

    def get_metric(self) -> int:
        return self.metric

    def normalize(self, tweet: str) -> str:
        return tweet


class DictionaryNormalizer(Normalizer):
    """Remove non-discriminatory stop token.

    Note that the metric is the number of token removed overall.
    """

    def __init__(self) -> None:
        super(DictionaryNormalizer, self).__init__()
        self.matchers_and_full_texts = []

        with open("normalization-dict.csv", "r") as f:
            for index, line in enumerate(f.readlines()):
                if index == 0:
                    # Skip the header line.
                    continue
                abbr, full = line.strip().split(",")
                self.matchers_and_full_texts.append(
                    (
                        re.compile(
                            f"(\s+{abbr}\s+)|(^{abbr}\s+)|(\s+{abbr}$)|(^{abbr}$)"
                        ),
                        f" {full} ",
                    )
                )

    def normalize(self, tweet: str) -> str:
        output = tweet
        for matcher, full in self.matchers_and_full_texts:
            self.metric += len(matcher.findall(output))
            output = matcher.sub(full, output)

        return output.strip()


dict_normalizer = DictionaryNormalizer()


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
    experiment.set_name("Dictionary Normalize (Train + Valid + Test)")

    # Need to provide the relative path to the folder containing the
    # training_set.csv and validation_set.csv
    path_to_dataset = os.path.join(os.pardir, os.pardir, "dataset")
    train, valid = load_train_valid_data(path_to_dataset)
    test = load_test_data(path_to_dataset)

    train["tweet"] = train["tweet"].apply(dict_normalizer.normalize)
    valid["tweet"] = valid["tweet"].apply(dict_normalizer.normalize)
    test["tweet"] = test["tweet"].apply(dict_normalizer.normalize)

    # Create a new subdirectory for normalized so it can easily be loaded later.
    if not os.path.exists("dataset-normalized"):
        os.makedirs("dataset-normalized")

    train.to_csv(os.path.join("dataset-normalized", "training_set.csv"))
    valid.to_csv(os.path.join("dataset-normalized", "validation_set.csv"))
    test.to_csv(os.path.join("dataset-normalized", "test_set.txt"), header=False)

    experiment.log_table(filename="training_set_normalized.csv", tabular_data=train)
    experiment.log_table(filename="validation_set_normalized.csv", tabular_data=valid)
    experiment.log_table(filename="test_set_normalized.csv", tabular_data=test)

    metrics = {"abbreviations_normalized": dict_normalizer.get_metric()}
    experiment.log_metrics(metrics)
    # Log metrics to local file system
    with open('metrics.yaml', 'w+') as outfile:
        yaml.dump(metrics, outfile)


if "__main__" == __name__:
    main()
