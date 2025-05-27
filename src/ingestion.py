import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd
import scipy.io as sio


def train_test_split(
    data: pd.DataFrame,
    test_size: Union[float, int] = 0.25,
    random_state: Union[int, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    X :  pd.DataFrame
         The input data to split.
    test_size : float, int, or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns:
    Tuple containing:
        - data_train: pd.DataFrame
        - data_test: pd.DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(data)
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    data_train = data.iloc[train_indices]
    data_test = data.iloc[test_indices]

    return data_train, data_test


def unpickle(file):
    """Load a pickled CIFAR-10 batch file."""
    with open(file, "rb") as fo:
        dict_data = pickle.load(fo, encoding="bytes")
    return dict_data


def process_data(
    data_dir: str, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CIFAR-10 dataset from extracted directory.

    Args:
    - data_dir (str): Path to extracted CIFAR-10 dataset directory.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and testing DataFrames.
    """
    batches = [f"{data_dir}/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_batch = f"{data_dir}/cifar-10-batches-py/test_batch"

    all_data, all_labels = [], []

    # Load training data
    for batch in batches:
        batch_data = unpickle(batch)
        all_data.append(batch_data[b"data"])
        all_labels.extend(batch_data[b"labels"])

    train_data = np.vstack(all_data).reshape(-1, 3, 32, 32).astype("float32") / 255.0
    train_labels = np.array(all_labels)

    # Load test data
    test_data_dict = unpickle(test_batch)
    test_data = test_data_dict[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    test_labels = np.array(test_data_dict[b"labels"])

    # Create DataFrames
    train_df = pd.DataFrame({"image": list(train_data), "label": train_labels})
    test_df = pd.DataFrame({"image": list(test_data), "label": test_labels})

    train_df, val_df = train_test_split(
        train_df,
        test_size=config.get("val_size", 0.2),
        random_state=config.get("random_state", 42),
    )
    logging.info(
        f"Prepared 3 data splits: train, size: {len(train_df)}, val: {len(val_df)}, test: {len(val_df)}"
    )
    return train_df, val_df, test_df
