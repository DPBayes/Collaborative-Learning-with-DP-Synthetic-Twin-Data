"""
Plots the downstream performance (measured in log-likelihood)
at different amounts of distributional skew from the global population in a center
using the results computed by compute_ll_over_number_of_centers.py.

Used to create Fig 5.
"""
import os, sys

import numpy as np
import pandas as pd

from collections import defaultdict

import argparse

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import filenamer

parser = argparse.ArgumentParser()
parser.add_argument("--test_data_path", type=str, default=None)
parser.add_argument("--output_dir", default="./", type=str)
parser.add_argument("--lls_path", default=None)
parser.add_argument("--epsilon", default=1.0, type=str, help="Privacy level")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=2.0, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--max_center_size", default=None, type=float, help="The maximum size of data points to consider from each center.")
parser.add_argument("--mc_seed", default=8126, type=int, help="Seed for randomness used in the script.")
parser.add_argument("--filter_extreme_lls", default=0., type=float, help="The cutoff percentage of extreme LLs do discard before plotting (i.e., a value of .2 will only plot mean and std over the 80 perc. confidence interval (.1 -> .9).")
parser.add_argument("--test_type", choices=["ll", "avg_ll"], default="avg_ll")
parser.add_argument("--annotate_significance", default=False, action='store_true')

args = parser.parse_args()

##### fit non-dp model
# read original data
orig_data_test = pd.read_csv(args.test_data_path)


# centers = list(orig_data_train["assessment_center"].unique())
fractions = [.1, .25, .5, .75, 1.]
centers = [("A" + str(fraction)) for fraction in fractions]

######################
## Read combined fits
epsilon = args.epsilon
n_epochs = args.num_epochs

normalization_factor = 1.
if args.test_type == "avg_ll":
    normalization_factor = 1. / len(orig_data_test)

import jax
rng_key = jax.random.PRNGKey(args.mc_seed)
orig_ll_rng_key, centers_ll_rng_key = jax.random.split(rng_key)

pred_ll_combined_dict = defaultdict(dict)
pred_ll_local_only_dict = defaultdict(dict)

for i, center in enumerate(centers):

    print(f"Processing center {center}")

    fname_lls = filenamer("lls_over_num_shared", center, args) + "_mc.p"
    fpath_lls = os.path.join(args.lls_path, fname_lls)
    lls = pd.read_pickle(fpath_lls)

    for i in range(10):
        seed = 123 + i
        pred_ll_local_only_dict[center][seed] = lls[i, 0, 0].ravel() * normalization_factor # samples for first position are copies of the local-only samples across all 'rep'; this will skew tests, so we just use the first

        pred_ll_combined_dict[center][seed] = lls[i, :, -1].ravel() * normalization_factor

## plot

pred_ll_local_only_df = pd.concat([
    pd.DataFrame(pred_ll_local_only_dict[center]).melt(value_name="lls", var_name="seed").assign(center=center) for center in pred_ll_local_only_dict
])

pred_ll_combined_df = pd.concat([
    pd.DataFrame(pred_ll_combined_dict[center]).melt(value_name="lls", var_name="seed").assign(center=center) for center in pred_ll_combined_dict
])

plot_df = pd.concat((
    pred_ll_combined_df.assign(variant="combined"),
    pred_ll_local_only_df.assign(variant="local only"),
))

import significance_tests
def compute_ranked_t_test_p(df: pd.DataFrame):
    combined_df = df[df.variant == "combined"]
    local_only_df = df[df.variant == "local only"]
    assert len(combined_df) > 0
    assert len(local_only_df) > 0

    return significance_tests.compute_ranked_t_test_p(combined_df, local_only_df, alternative="two-sided")

p_vals = plot_df.groupby("center").apply(compute_ranked_t_test_p)
p_vals.index = fractions
fraction_labels = fractions
if args.annotate_significance:
    fraction_labels = significance_tests.annotate_significance(p_vals, fractions)

print("#### p values for ranked Welch t-test ####")
print(p_vals)

import matplotlib.pyplot as plt
import seaborn as sns

from fig_style import get_fig_style
plt.rcParams.update(get_fig_style())

fig, axis = plt.subplots()

skew_name = "South Asian"

sns.boxplot(data=plot_df, x='center', y='lls', hue='variant', order=centers, ax=axis, showfliers=False, linewidth=.5, width=.5)
axis.grid(axis='x', ls="-.", lw=".1", c="grey", alpha=.5)
axis.set_xticks(np.arange(len(fractions)))
axis.set_xticklabels(fraction_labels)
axis.set_ylabel(f"Test log-likelihood, {skew_name} only")
axis.set_xlabel(f"Subsampling ratio for ({skew_name}, positive) marginal")
axis.set_title(f"Large center improvement depending on local data skew \n {skew_name}, $\epsilon={epsilon}$")
axis.legend()

max_size_suffix = ""
if args.max_center_size is not None:
    max_size_suffix = f"_max{args.max_center_size}"
plt.savefig(
    os.path.join(
        args.output_dir,
        f"{args.test_type}_skewed{skew_name.replace(' ', '_')}_test_loglikelihood_eps{epsilon}_ne{n_epochs}_80_20_split{max_size_suffix}.pdf"
    ),
    format="pdf", bbox_inches="tight")
plt.close()
