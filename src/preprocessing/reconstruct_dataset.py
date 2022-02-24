import os
import sys
import pandas as pd


sys.path.append("../data_processing")
from loading import load_train_valid_data, load_test_data

def main():

  # Load tweets
  path_to_dataset = os.path.join(os.pardir, os.pardir, "dataset")
  train, valid = load_train_valid_data(path_to_dataset)
  test = load_test_data(path_to_dataset)

  # Load tweets with subwords
  train_subwords = pd.read_csv('training_set_subwords_text_only.csv', sep='\t\t\t')
  valid_subwords = pd.read_csv('validation_set_subwords_text_only.csv', sep='\t\t\t')
  test_subwords = pd.read_csv('test_set_subwords_text_only.csv', sep='\t\t\t')

  test_subwords.index += 1


  # Merge tweets with subwords and labels
  train_joined = train.join(train_subwords, lsuffix='_orig', rsuffix='_subwords')
  train_joined['tweet'] = train_joined['tweet_subwords'].apply(lambda tweet: tweet.strip('"'))

  del train_joined['tweet_orig']
  del train_joined['tweet_subwords']

  train_updated = train_joined[['tweet', 'label']]

  train_updated.to_csv(os.path.join('dataset-subwords', 'training_set.csv'))


  valid_joined = valid.join(valid_subwords, lsuffix='_orig', rsuffix='_subwords')
  valid_joined['tweet'] = valid_joined['tweet_subwords'].apply(lambda tweet: tweet.strip('"'))

  del valid_joined['tweet_orig']
  del valid_joined['tweet_subwords']

  valid_updated = valid_joined[['tweet', 'label']]

  valid_updated.to_csv(os.path.join('dataset-subwords', 'validation_set.csv'))


  test_joined = test.join(test_subwords, lsuffix='_orig', rsuffix='_subwords')
  test_joined['tweet'] = test_joined['tweet_subwords'].apply(lambda tweet: tweet.strip('"'))

  del test_joined['tweet_orig']
  del test_joined['tweet_subwords']

  test_joined.to_csv(os.path.join('dataset-subwords', 'test_set.txt'))


if __name__ == "__main__":
    main()