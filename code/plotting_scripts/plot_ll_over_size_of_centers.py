"""
Plots the downstream performance (measured in log-likelihood)
at a single center for different sizes of local data available to centers
using the results computed by compute_ll_over_number_of_centers.py.

Used to create Fig 3.
"""
import os, sys

import numpy as np
import pandas as pd

import argparse

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import filenamer

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("center", type=str, help="The center for which to plot the improvement")
parser.add_argument("--train_data_path", default=None)
parser.add_argument("--test_data_path", default=None)
parser.add_argument("--lls_path", default=None)
parser.add_argument("--output_path", default="./")
parser.add_argument("--epsilon", default=1.0, type=str, help="Privacy level")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=2.0, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--test_type", choices=["ll", "avg_ll"], default="avg_ll")
parser.add_argument("--plot_variant", choices=["single_box", "single_violin", "single_line"], default="single_box")
parser.add_argument("--annotate_significance", default=False, action='store_true')

args = parser.parse_args()
print(args)

##### fit non-dp model
# read original data
orig_data_test = pd.read_csv(args.test_data_path, index_col=0)

normalization_factor = 1.
if args.test_type == "avg_ll":
    normalization_factor = 1. / len(orig_data_test)

orig_data_train = pd.read_csv(args.train_data_path, index_col=0)
del orig_data_train["covid_test_present"]
del orig_data_train["covid_test_positive_in_hospital"]

# population level fit
from utils import fit_model1, pred_test_likelihood
orig_pop_level_fit = fit_model1(orig_data_train)[0]
pred_ll_orig_wholepop = pred_test_likelihood(orig_pop_level_fit.params, orig_data_test, return_average=False) * normalization_factor

center = args.center

epsilon = args.epsilon
n_epochs = args.num_epochs

def load_center_lls(center_name, center_size):
    max_center_size = center_size
    if center_size == 1.0:
        max_center_size = None
    fname_output = filenamer("lls_over_num_shared", center_name, args, seed=None, max_center_size=max_center_size) + "_mc.p"
    fpath_output = os.path.join(args.lls_path, fname_output)
    lls = pd.read_pickle(fpath_output)
    return lls

def lls_to_df(lls, center_size):
    num_starts, num_seeds, num_repetitions, seq_length, num_mc = lls.shape

    seq_length = 1
    local_lls_flat = np.ravel(lls[:, :, :, 0]) * normalization_factor
    combined_lls_flat = np.ravel(lls[:, :, :, -1]) * normalization_factor
    lls_start = np.repeat(start_centers, num_seeds * num_repetitions * seq_length * num_mc)
    lls_seeds = np.tile(np.repeat(np.arange(num_seeds), num_repetitions * seq_length * num_mc), num_starts)
    lls_reps = np.tile(np.repeat(np.arange(num_repetitions), seq_length * num_mc), num_starts * num_seeds)
    
    lls_df = pd.concat((
        pd.DataFrame({"lls": combined_lls_flat, "local center": lls_start, "seed": lls_seeds, "rep": lls_reps, "variant": "combined", "center_size": center_size}),
        pd.DataFrame({"lls": local_lls_flat, "local center": lls_start, "seed": lls_seeds, "rep": lls_reps, "variant": "local only", "center_size": center_size}),
    ))

    return lls_df

if center == "all":
    with open("../all_centers.txt", "r") as f:
        start_centers = [c.strip() for c in f.readlines()]
else:
    start_centers = [center]

#center_sizes = [0.1, 0.2, 0.5, 1.0]
center_sizes = [0.1, 0.2, 0.5, 1.0]
lls_df = pd.DataFrame()
for center_size in center_sizes:
    lls = np.array([load_center_lls(center, center_size) for center in start_centers])
    assert len(lls.shape) == 5
    assert lls.shape[-2] == 16

    lls_df = pd.concat((lls_df, lls_to_df(lls, center_size)), axis=0)


import significance_tests
def compute_ranked_t_test_p(df: pd.DataFrame):
    first_df = df[df.variant == "combined"]
    second_df = df[df.variant == "local only"].groupby("seed").apply(lambda seed_df: seed_df[seed_df.rep == 0]) # samples for first position are copies of the local-only samples across all 'rep'; this will skew tests, so we just use the first

    assert len(first_df) > 0
    assert len(second_df) > 0

    return significance_tests.compute_ranked_t_test_p(first_df, second_df)

p_vals = lls_df.groupby("center_size").apply(compute_ranked_t_test_p)
center_size_labels = center_sizes
if args.annotate_significance:
    center_size_labels = significance_tests.annotate_significance(p_vals, center_sizes)

print("#### p values for ranked Welch t-test ####")
print(p_vals)

from fig_style import get_fig_style
import seaborn as sns

box_plot_regions = [(-1.8, -.45), (-4.3, -3.1)]
box_plot_region_heights = [reg[1] - reg[0] for reg in box_plot_regions]


def plot_fn(data, *, hue=None, ax=None, hue_order=None):
    ax.axhline(pred_ll_orig_wholepop, ls="--", color="k", label='full population')

    if args.plot_variant.endswith("box"):
        sns.boxplot(data=data, x='center_size', y='lls', hue=hue, hue_order=hue_order, ax=ax, showfliers=False, linewidth=.5, width=.5)
        ax.grid(axis='x', ls="-.", lw=".1", c="grey", alpha=.5)

    elif args.plot_variant.endswith("violin"):
        sns.violinplot(data=data, x='center_size', y='lls', hue=hue, hue_order=hue_order, ax=ax, split=(len(data[hue].unique()) == 2), cut=0, scale="count", gridsize=1000)
        ax.set_ylim(-4., -0.5)
    elif args.plot_variant.endswith("line"):
        sns.lineplot(data=data, x='center_size', y='lls', hue=hue, hue_order=hue_order, ax=ax, err_style="bars", errorbar="ci", estimator="median", alpha=.6)

if args.plot_variant.startswith("single"):
    plt.rcParams.update(get_fig_style())
    fig, ax = plt.subplots()

    plot_fn(lls_df, hue='variant', ax=ax)

    ax.set_xlabel("Fraction of full local data set")
    ax.set_ylabel("Test log-likelihood")
    ax.set_xticklabels(center_size_labels)
    ax.legend()

    if len(lls_df['variant'].unique()) < 2:
        ax.get_legend().remove()

    fig.subplots_adjust(left=.12, right=.98)

if center == "all":
    fig.suptitle(f"Smaller local data sets result in larger improvement\n Test log-likelihoods for all centers, $\epsilon = {epsilon}$")
else:
    fig.suptitle(f"Smaller local data sets result in larger improvement\n {center}, $\epsilon = {epsilon}$")

fig.savefig(os.path.join(args.output_path, f"{args.test_type}_over_local_size_{center}_mc_{args.plot_variant}.pdf"), format="pdf")
plt.close()

