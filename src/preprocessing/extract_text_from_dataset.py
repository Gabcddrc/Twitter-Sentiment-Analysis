import os
import sys

sys.path.append("../data_processing")
from loading import load_train_valid_data, load_test_data


def tweetNormalizer(sentence):
    sentence = sentence.replace("<url>", "HTTPURL")
    sentence = sentence.replace("<user>", "@USER")
    return sentence


def main():
    path_to_dataset = os.path.join(os.pardir, os.pardir, "dataset")
    train, valid = load_train_valid_data(path_to_dataset)
    test = load_test_data(path_to_dataset)

    train["tweet"] = train["tweet"].apply(tweetNormalizer)
    valid["tweet"] = valid["tweet"].apply(tweetNormalizer)
    test["tweet"] = test["tweet"].apply(tweetNormalizer)

    train["tweet"].to_csv("training_set_text_only.csv", index=False, header=False)
    valid["tweet"].to_csv("validation_set_text_only.csv", index=False, header=False)
    test["tweet"].to_csv("test_set_text_only.csv", index=False, header=False)


if __name__ == "__main__":
    main()
