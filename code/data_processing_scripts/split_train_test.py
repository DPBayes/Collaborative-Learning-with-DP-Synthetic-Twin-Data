""" splits the preprocessed data CSV into train/test sets for a given test set size. splits are always stratified by center and optionally stratified by another feature within each center."""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Optional
import os
import sys
sys.path.append("../")
import paths

parser = argparse.ArgumentParser()
parser.add_argument("--test-data-pct", type=int, default=20, help="Size of the test data set to be split off from the input data; in percent. Default=20.")
parser.add_argument("--stratify", type=str, default="ethnicity", help="Dimension on which to stratify the split (or 'None' to perform no stratification). Default=ethnicty.")
parser.add_argument("--seed", type=int, default=22812563)
args = parser.parse_args()

test_suffix = f"{args.test_data_pct:03}"
train_suffix = f"{(100 - args.test_data_pct):03}"
test_split_ratio = args.test_data_pct / 100.
print(f"Splitting data into {(100 - args.test_data_pct)}% train and {args.test_data_pct}% test sets...")

rs = np.random.RandomState(args.seed)

def split_dataset(df: pd.DataFrame) -> None:
    stratify = df[args.stratify] if args.stratify.lower().strip() != "none" else None
    try:
        train_df, test_df = train_test_split(df, test_size=test_split_ratio, stratify=stratify, random_state=rs)
    except ValueError:
        print("!! Could not stratify due to too small class; falling back to unstratified splitting !!")
        train_df, test_df = train_test_split(df, test_size=test_split_ratio, stratify=None, random_state=rs)

    # train_df.to_csv(os.path.join(out_dir_path, f"{in_file_base_name}_train{train_suffix}.csv"))
    # test_df.to_csv(os.path.join(out_dir_path, f"{in_file_base_name}_test{test_suffix}.csv"))
    return train_df, test_df

file_path = paths.model_one_covid_tested_data
df_whole = pd.read_csv(file_path, index_col=0)
centers = list(df_whole["assessment_center"].unique())

# we construct train/test split for whole data by concatenating the splits of the per-center sets
# to ensure that test and train splits are consistent in both views
wholepop_train_df, wholepop_test_df = pd.DataFrame(), pd.DataFrame()
for center in centers: # split each center
    print(f"{center}..")
    df_center = df_whole[df_whole["assessment_center"] == center]
    center_train_df, center_test_df = split_dataset(
        df_center
    )

    # concatenate for center split for wholepop
    wholepop_train_df = pd.concat((wholepop_train_df, center_train_df))
    wholepop_test_df = pd.concat((wholepop_test_df, center_test_df))

print("Building wholepop train/test datasets by re-combining centers")
wholepop_train_df.to_csv(os.path.join(paths.processed_data_path, f"model_one_covid_tested_data_train{train_suffix}.csv"))
wholepop_test_df.to_csv(os.path.join(paths.processed_data_path, f"model_one_covid_tested_data_test{test_suffix}.csv"))

print("Done")
