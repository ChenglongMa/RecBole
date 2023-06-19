#!/usr/bin/env python
# coding: utf-8

import math
import os
from typing import Union

import numpy as np
import pandas as pd


def process_split_ratio(ratio: Union[float, list, tuple]) -> list:
    """
    Generate split ratio lists.
    :param ratio: (float or list or tuple): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).
    :return: a list of normalized split ratio.
    """
    if isinstance(ratio, float):
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Split ratio has to be between 0 and 1")
        ratio = [ratio, 1 - ratio]
    else:
        if any([x <= 0 for x in ratio]):
            raise ValueError(
                "All split ratios in the ratio list should be larger than 0."
            )

        # normalize split ratios if they are not summed to 1
        if math.fsum(ratio) != 1.0:
            ratio = [x / math.fsum(ratio) for x in ratio]
    return ratio


def split_by_ratios(data: pd.DataFrame, ratios: Union[list, tuple], shuffle=False, order_by: str = None, ascending=True, seed: int = 42):
    """
    Helper function to split pandas DataFrame with given ratios
    :param data: (pandas.DataFrame): Pandas data frame to be split.
    :param ratios: (list of floats): list of ratios for split. The ratios have to sum to 1.
    :param shuffle: (bool): whether data will be shuffled when being split.
    :param order_by: column name which is used to sort the data
    :param ascending: sort order
    :param seed: (int): random seed.
    :return:
    """
    split_index = np.cumsum(ratios).tolist()[:-1]
    if shuffle and order_by:
        raise ValueError(
            '"shuffle" cannot be True when "order_by" is specified.')
    if shuffle:
        data = data.sample(frac=1, random_state=seed)
    if order_by:
        data = data.sort_values(by=order_by, ascending=ascending)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # # Add split index (this makes splitting by group more efficient).
    # for i in range(len(ratios)):
    #     splits[i]["split_index"] = i

    return splits


def split_data(full_data: pd.DataFrame, by: str = None, order_by=None, ascending=True, ratio: Union[float, list, tuple] = .7, seed=42) -> list[
    pd.DataFrame]:
    """
    Split data into training set and test set
    :param full_data: The full data set
    :type full_data: pd.DataFrame
    :param by: the key column to split, can be None, 'user' or 'item', defaults to None (randomly splitting)
    :type by: str
    :param order_by: the column to keep order in resulting datasets, defaults to None, e.g., 'timestamp'
    :type order_by: str
    :param ascending:
    :param ratio:
    :param seed:
    :return: a list of pd.DataFrame, len(result) == len(ratios)
    :rtype: list[pd.DataFrame]
    """
    ratios = process_split_ratio(ratio)

    if by is None:
        # split the data randomly
        splits = split_by_ratios(full_data, ratios, shuffle=order_by is None, order_by=order_by, seed=seed)
    else:
        if order_by is None:
            full_data['random'] = np.random.default_rng(seed).random(size=full_data.shape[0])
            order_by = 'random'
        groups = full_data.groupby(by=by)
        rank = groups[order_by].rank(method='dense', pct=True, ascending=ascending)
        split_index = [0] + np.cumsum(ratios).tolist()
        if 'random' in full_data.columns:
            full_data = full_data.drop('random', axis=1)
        splits = [full_data[rank.between(split_index[x - 1], split_index[x])].copy() for x in range(1, len(split_index))]
    return splits


if __name__ == '__main__':
    dir_name = './dataset'
    dataset_name = 'ml-100k'
    target_dataset_name = 'ml-100k-split'

    filename = f'{dataset_name}.inter'
    sep = '\t'
    ratio = [0.8, 0.1, 0.1]
    random_state = 42
    header = 0
    columns = 'user_id:token item_id:token rating:float timestamp:float'.split()

    df = pd.read_csv(f'{dir_name}/{dataset_name}/{filename}', header=header, sep=sep)
    df_train, df_valid, df_test = split_data(df, by='user_id:token', order_by='timestamp:float', ratio=ratio, seed=random_state)
    df_train['phase:token'] = 'train'
    print(df_train.shape)
    df_valid['phase:token'] = 'valid'
    print(df_valid.shape)
    df_test['phase:token'] = 'test'
    print(df_test.shape)

    dataset_name = target_dataset_name
    os.makedirs(f'{dir_name}/{dataset_name}', exist_ok=True)

    df_train.to_csv(f'{dir_name}/{dataset_name}/{dataset_name}.train.inter', index=False)
    df_valid.to_csv(f'{dir_name}/{dataset_name}/{dataset_name}.valid.inter', index=False)
    df_test.to_csv(f'{dir_name}/{dataset_name}/{dataset_name}.test.inter', index=False)
    print('Done')
