import os
import pandas as pd

def load_train_valid_data(dataset_path):
    """Load deduped training and validation data.

    Args:
        dataset_path (str): Path to 'dataset' folder.

    Returns:
        Tuple of pandas DataFrame: First element is the training data, second
            the validation data.
    """
    train = pd.read_csv(
        os.path.join(dataset_path, 'training_set.csv'),
        index_col = 'id'
    )
    validate = pd.read_csv(
        os.path.join(dataset_path, 'validation_set.csv'),
        index_col = 'id'
    )
    return train, validate 

def load_test_data(dataset_path):
    """Load test set.

    Args:
        dataset_path (str): Path to 'dataset' folder.

    Returns:
        pandas.DataFrame: Row index corresponds with tweet index (N.B.: starts
            with 1!), single column with tweets.
    """
    with open(os.path.join(dataset_path, 'test_set.txt')) as f:
        data = [line.split(',', maxsplit = 1) for line in f]

    # Remove trailing newline.
    index = [int(id) for id, tweet in data]
    tweets = [tweet[:-1] for id, tweet in data]

    test = pd.DataFrame(tweets, index = index, columns = ['tweet'])
    # For consistency with training and validation data.
    test.index.name = 'id'

    return test