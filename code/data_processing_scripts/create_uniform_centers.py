import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser("splits UKB data into uniformly distributed centers")
parser.add_argument("data_path", type=str)
parser.add_argument("out_data_path", type=str)
parser.add_argument("num_centers", type=int)
parser.add_argument("--seed", type=int, default=7623)
parser.add_argument("--separate_outputs", default=False, action='store_true')

args = parser.parse_args()

# center_names = start_names[:args.num_centers]
center_names = [chr(ord('A') + i) for i in range(args.num_centers)]

df_whole = pd.read_csv(args.data_path)
df_whole.drop(columns=["assessment_center"])

rs = np.random.RandomState(args.seed)
center_size = (len(df_whole) // args.num_centers) + 1
center_assignments = rs.permutation(np.repeat(center_names, center_size))[:len(df_whole)]
df_whole['assessment_center'] = center_assignments

if args.separate_outputs:
    for center in center_names:
        center_df = df_whole[df_whole['assessment_center'] == center]
        center_df.to_csv(args.out_data_path + "." + center, index=False)
else:
    df_whole.to_csv(args.out_data_path, index=False)
