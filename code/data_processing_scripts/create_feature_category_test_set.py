import pandas as pd
import argparse


parser = argparse.ArgumentParser("reduces test set to a particular feature category (subgroup)")
parser.add_argument("data_path", type=str)
parser.add_argument("out_data_path", type=str)
parser.add_argument("--feature", type=str, default="ethnicity", help="The feature from which a category will be subsampled.")
parser.add_argument("--category", type=str, default="South Asian", help="The category to subsample.")
parser.add_argument("--center", type=str, default=None, help="Only apply to the selected center (ignoring all data from other centers).")

args = parser.parse_args()

df_whole = pd.read_csv(args.data_path)

centers = list(df_whole['assessment_center'].unique())
if args.center is not None:
    centers = [args.center]

new_df_whole = pd.DataFrame()
for center in centers:
    center_df = df_whole[df_whole['assessment_center'] == center]

    category_df = center_df[center_df[args.feature] == args.category]
    
    new_df_whole = pd.concat((new_df_whole, category_df))

new_df_whole.to_csv(args.out_data_path, index=False)
