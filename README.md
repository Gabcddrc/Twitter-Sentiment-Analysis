# Preparing the Dataset

We have removed duplicate tweets from the original dataset and split the training
set into training and validation. For reproducibility, we have included the
modified dataset within this repository.

**_Before running any experiments_**, you will need to unzip the `dataset.zip` in
the root of this repository:

```
unzip dataset.zip
```

Afterwards, make sure your directory structure resembles:

```
.
|-- README.md
|-- LICENSE
|-- src/
|-- dataset/
|   |-- test_set.txt
|   |-- training_set.csv
|   `-- validation_set.csv
`-- dataset.zip
```

All code provided herein relies on the `dataset` folder being in the root of this repository, on the same level as the `src` folder.

# Replicating the Experiments

## Baselines

The baseline experiments for logistic regression, BERT, and BERTweet can by found in `src/baselines`. The corresponding experiments to reproduce the numbers from the report are `log_reg.py`, `bert.py`, and `bertweet.py`.
Running the GloVe experiment is more involved.
### GloVe
Before running the GloVe experiment you need to add pre-trained GloVe embeddings to the project as follows:
1. Download the embeddings from https://nlp.stanford.edu/data/glove.twitter.27B.zip.
2. Unzip `glove.twitter.27B.zip`.
3. Copy `glove.twitter.27B.200d.txt` to the project folder at the path `embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt`.

To train and evaluate the sentiment classifier that fits a neural network to the average GloVe embedding run
```
cd src/glove && python glove_average_mlp.py
```

## BERTweet with Weight Freezing
The experiments applying weight freezing for fine-tuning BERTweet can be found in `src/bertweet`. To run the BERTweet model with a single final layer execute `bertweet_freezing.py` for extended model architecture use `bertweet_multiout.py`. To train extended architecture on the whole dataset, which was used for the Kaggle competition, use `bertweet_submission.py`.

## Preprocessing

Generally, every preprocessing experiment can be run with either Logistic Regression or BERTweet. For simplicity, we only provide the commands for running the Logistic Regression. To run the BERTweet experiments, replace every `log_reg_*` with `bertweet_*`. 

Note that every preprocessing experiment is placed within the `src/preprocessing` folder, so change to this folder before executing any further commands. Some scripts generate files within their execution directory and other scripts depend on these or other files within `src/preprocessing`, so will fail if not executed in the correct directory. If currently in the root directory of the repository, use the following command to change to the correct directory:

```
cd src/preprocessing
```

The output of the actual experiments is always a CSV file containing the test set predictions and a `metrics.yaml` containing training metrics, including validation accuracy as `valid_score`.

### Punctuation Removal

To run the punctuation removal experiment, simply execute:

```
python log_reg_punctuation_removal.py
```

### Stop Word Removal

Before running the actual experiment, we need to generate the list of stop words to be removed. The first step is to generate a count of positive occurrences and negative occurrences of every token in the dataset:

```
python count_token.py
```

After this, we need to generate the list of stop words based on the counts:

```
python stop_token.py
```

Finally, run the stop word experiment:

```
python log_reg_stop_token_removal.py
```

### Dictionary Normalization

Before running the actual experiment, we need to generate a normalized version
of the dataset:

```
python dictionary_normalize.py
```

This will create a folder `dataset-normalized` that is used by the dictionary
normalization experiments. To now run the experiment, simply execute:

```
python log_reg__dict_normalization.py 
```

### Subword Units

This experiment requires the additional python package [`subword-nmt`](https://github.com/rsennrich/subword-nmt) to be installed. To install the package, follow the [package's installation instructions](https://github.com/rsennrich/subword-nmt).

After `subword-nmt` is installed, we need to train the subword units and generate a dataset where they have been applied. For convenience, we have grouped all necessary commands together in the script `create_subword_dataset.sh`. Therefore, simply run:

```
./create_subword_dataset.sh
```

If not on a Linux-like environment, simply execute each command from the script in the same order as in the script. This will create a `dataset-subwords` that is
used by the subword units experiments.

Finally, run the subword unit experiment:

```
python log_reg_subword_units.py
```

## Analyses

Outside of the baseline and preprocessing experiments, we have conducted
analyses of mislassifications by BERTweet and of the dataset in general. These can
be replicated via the following instructions.

### BERTweet Misclassification

Navigate to the `analysis/misclassification` folder within the repository. If
currently in the root directory of the repository, use the following command to
change to the correct directory:

```
cd analysis/misclassification
```

Now execute the `bertweet.ipynb` Jupyter Notebook and follow along.

### Translation

Within this analysis we explore the non-English tweets contained within our
dataset. We use the Google Translate API to detect the language of a given tweet
and translate it if necessary.

Navigate to the `analysis/translation` folder within the repository. If
currently in the root directory of the repository, use the following command to
change to the correct directory:

```
cd analysis/translation
```

The `translation.ipynb` Jupyter Notebook contains all the necessary code to
replicate the analysis.

Note that in order to run the entire code, you will need to
[setup the Google Translate API for Python](https://codelabs.developers.google.com/codelabs/cloud-translation-python3#0). However, for your convenience, we have precomputed
all files that would require the Google Translate API and by default the Jupyter Notebook will use the precomputed files. Therefore, you can still follow
along with the Jupyter Notebook, even without setting up the API or changing
any code.

If you have setup the Google Translate API and prefer to recompute the results,
you simply need to set the `USE_PRECOMPUTED` variable to `False` in the first
code cell of the Jupyter Notebook and then follow along normally.

# Kaggle Competition Submission
For the final submission to the Kaggle competition we train our BERTweet model (with layer freezing and two hidden layers for the final classifier) on the full dataset using
```
cd bertweet && python bertweet_submission.py
```
which saves the prediction file to
```
bertweet2_multiout2_full_prediction.csv
```
# Troubleshooting

A common cause for failing experiments can be the incorrect placement of the `dataset` folder. All experiments rely on a relative path to this folder, please make sure to follow the instructions above on _Preparing Dataset_ correctly.