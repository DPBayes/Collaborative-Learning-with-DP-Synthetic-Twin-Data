import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser("creates new center local training sets of smaller sizes by subsampling the full UKB data")
parser.add_argument("data_path", type=str)
parser.add_argument("out_data_path", type=str)
parser.add_argument("center_size", type=float, help="The size of the subsampled data set; if <=1, interpreted as a fraction of the original data size, if integer > 1, interpreted as absolute size")
parser.add_argument("--seed", type=int, default=9836532)

args = parser.parse_args()

df_whole = pd.read_csv(args.data_path)
centers = df_whole['assessment_center'].unique()

rs = np.random.RandomState(args.seed)

sampled_dfs = []
for center in centers:
    df_center = df_whole[df_whole["assessment_center"] == center]

    new_center_size = args.center_size
    if new_center_size < 1.:
        new_center_size = len(df_center) * new_center_size
    new_center_size = int(new_center_size)
    idxs = rs.choice(len(df_center), new_center_size, replace=False)
    sampled_dfs.append(df_center.iloc[idxs])

df_sampled = pd.concat(sampled_dfs)
df_sampled.to_csv(args.out_data_path, index=False)
