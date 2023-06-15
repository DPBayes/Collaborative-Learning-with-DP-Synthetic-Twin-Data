import pandas as pd
import argparse
import numpy as np

fractions = [0., .1, .25, .5, .75, 1.]

parser = argparse.ArgumentParser("creates new centers in input data by subsampling different fractions of a specific feature category")
parser.add_argument("data_path", type=str)
parser.add_argument("out_data_path", type=str)
parser.add_argument("--feature", type=str, default="ethnicity", help="The feature from which a category will be subsampled.")
parser.add_argument("--category", type=str, default="South Asian", help="The category to subsample.")
parser.add_argument("--center", type=str, default=None, help="Only apply to the selected center (ignoring all data from other centers).")
parser.add_argument("--seed", type=int, default=147312)

args = parser.parse_args()

df_whole = pd.read_csv(args.data_path)

centers = list(df_whole['assessment_center'].unique())
if args.center is not None:
    centers = [args.center]

rs = np.random.RandomState(args.seed)

new_df_whole = pd.DataFrame()
for center in centers:
    center_df = df_whole[df_whole['assessment_center'] == center]

    marginal_filter = (center_df[args.feature] == args.category) & (center_df['covid_test_result'] == True)
    marginal_df = center_df[marginal_filter]
    without_category_df = center_df[~marginal_filter]

    assert fractions[-1] == 1.
    sampled_dfs = {1.: marginal_df}
    for fraction, prev_fraction in zip(reversed(fractions[:-1]), reversed(fractions[1:])):
        frac_of_prev = fraction / prev_fraction
        assert frac_of_prev < 1.

        sampled_df = sampled_dfs[prev_fraction].sample(frac=frac_of_prev, replace=False, random_state=rs)
        sampled_dfs[fraction] = sampled_df

    for fraction, sampled_df in sampled_dfs.items():
        sampled_dfs[fraction] = pd.concat((without_category_df, sampled_df))\
            .assign(assessment_center=center + str(fraction))\
            .sample(frac = 1., random_state=rs) # shuffles data to mix categories

    new_df_whole = pd.concat((new_df_whole, *sampled_dfs.values()))

new_df_whole.to_csv(args.out_data_path, index=False)
