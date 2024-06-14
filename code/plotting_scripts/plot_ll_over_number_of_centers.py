"""
Plots the evolution of downstream performance (measured in log-likelihood)
at a single center over the number of available synthetic data sets from other centers
using the results computed by compute_ll_over_number_of_centers.py.

Used to create Fig 2.
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
parser.add_argument("--max_center_size", default=None, type=float, help="The maximum size of data points to consider from each center.")
parser.add_argument("--test_type", choices=["ll", "avg_ll"], default="avg_ll")
parser.add_argument("--plot_variant", choices=["single_box", "min_and_max_box", "all_separate_box"], default="min_and_max_box")
parser.add_argument("--plot_without_barts", default=False, action="store_true")
parser.add_argument("--annotate_significance", default=False, action="store_true")

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

def load_center_lls(center_name):
    fname_output = filenamer("lls_over_num_shared", center_name, args, seed=None) + "_mc.p"
    fpath_output = os.path.join(args.lls_path, fname_output)
    lls = pd.read_pickle(fpath_output)
    return lls

def lls_to_df(lls):
    num_starts, num_seeds, num_repetitions, seq_length, num_mc = lls.shape

    # start_center, seeds, reps, centers, samples
    lls_flat = np.ravel(lls) * normalization_factor
    lls_start = np.repeat(start_centers, num_seeds * num_repetitions * seq_length * num_mc)
    lls_seeds = np.tile(np.repeat(np.arange(num_seeds), num_repetitions * seq_length * num_mc), num_starts)
    lls_reps = np.tile(np.repeat(np.arange(num_repetitions), seq_length * num_mc), num_starts * num_seeds)
    lls_pos = np.tile(np.repeat(np.arange(seq_length), num_mc), num_starts * num_seeds * num_repetitions)

    lls_df = pd.DataFrame({"lls": lls_flat, "local center": lls_start, "seed": lls_seeds, "rep": lls_reps, "pos": lls_pos})

    return lls_df

if center == "all":
    start_centers = list(np.unique(orig_data_train["assessment_center"]))
else:
    start_centers = [center]

lls = np.array([load_center_lls(center) for center in start_centers])
assert len(lls.shape) == 5
assert lls.shape[-2] == 16

lls_df = lls_to_df(lls)

import significance_tests
def compute_two_pos_ranked_t_test_p(center_df: pd.DataFrame, first_pos: int, second_pos: int = None):
    if second_pos is None:
        second_pos = first_pos + 1
    first_df = center_df[center_df.pos == first_pos]
    if first_pos == 0: # samples for first position are copies of the local-only samples across all 'rep'; this will skew tests, so we just use the first
        first_df = first_df.groupby("seed").apply(lambda df: df[df.rep == 0])
    second_df = center_df[center_df.pos == second_pos]
    if len(first_df) == 0 or len(second_df) == 0:
        raise ValueError(f"no samples are present for one of the position indices ({first_pos=}, {second_pos=})")

    return significance_tests.compute_ranked_t_test_p(second_df, first_df)

def compute_ranked_t_test_p(center_df: pd.DataFrame):
    max_pos = center_df.pos.max()
    num_pos = max_pos + 1
    return pd.Series([
            compute_two_pos_ranked_t_test_p(center_df, i) for i in range(num_pos - 1)
        ],
        index=[f"{i} v {i+1}" for i in range(num_pos - 1)]
    )

p_vals = lls_df.groupby("local center").apply(compute_ranked_t_test_p)

from fig_style import get_fig_style
import seaborn as sns


def plot_fn(data, *, hue=None, ax=None, hue_order=None, width=0.5):
    ax.axhline(pred_ll_orig_wholepop, ls="--", color="k", label='full population')
    sns.boxplot(data=data, x='pos', y='lls', hue=hue, hue_order=hue_order, ax=ax, showfliers=False, linewidth=.5, width=width)
    ax.grid(axis='x', ls="-.", lw=".1", c="grey", alpha=.5)


if args.plot_variant.startswith("single"):
    plt.rcParams.update(get_fig_style())
    fig, ax = plt.subplots()

    plot_fn(lls_df, hue='variant', ax=ax)

    if len(lls_df['variant'].unique()) < 2:
        ax.get_legend().remove()

    fig.subplots_adjust(left=.12, right=.98)

elif args.plot_variant in ("min_and_max_box", "all_separate_box"):
    plot_df = lls_df
    hue_order = None
    if args.plot_variant == "min_and_max_box":
        min_med_max_centers = ("Barts", "Sheffield", "Leeds")
        hue_order = min_med_max_centers
        plot_df = lls_df[np.logical_or.reduce([lls_df['local center'] == c for c in min_med_max_centers])]

    ax2 = None
    if args.plot_variant == "all_separate_box":

        box_plot_regions = [(-1.8, -.5), (-4.3, -3.1)]
        box_plot_region_heights = [reg[1] - reg[0] for reg in box_plot_regions]

        plt.rcParams.update(get_fig_style(aspect_ratio=1.414, width_scale=2*1.414)) #12/12, scale=2
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': box_plot_region_heights})
        plot_width = .7

    else:

        box_plot_regions = [(-1.2, -.5), (-4.3, -3.1)]
        box_plot_region_heights = [reg[1] - reg[0] for reg in box_plot_regions]

        plot_width = .5

        if args.plot_without_barts:
            plt.rcParams.update(get_fig_style(aspect_ratio=12/9))
            fig, ax1 = plt.subplots(1, 1)
        else:
            plt.rcParams.update(get_fig_style(aspect_ratio=12/12))
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': box_plot_region_heights})

    plot_fn(plot_df, hue='local center', ax=ax1, hue_order=hue_order, width=plot_width)
    ax1.set_ylim(box_plot_regions[0])
    ax1.set_ylabel(None)
    ax1.set_xlabel(None)
    lowest_ax = ax1

    if ax2 is not None:
        plot_fn(plot_df, hue='local center', ax=ax2, hue_order=hue_order, width=plot_width)
        ax2.set_ylim(box_plot_regions[1])
        lowest_ax = ax2


        # hide the spines between ax and ax2
        ax1.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        ax1.get_legend().remove()

        ax2.set_xlabel(None)
        ax2.set_ylabel(None)
        ax2.legend(loc='lower right')

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        fig.subplots_adjust(hspace=.05)
        lowest_ax = ax2

    if args.plot_variant == "all_separate_box":
        # ax.set_ylim(-4., -0.5)
        ax2.legend(ncol=4, loc='lower right')
        fig.subplots_adjust(left=.05, right=.98, bottom=.05, top=.95)
    else:
        lowest_ax.legend()
        fig.subplots_adjust(left=.14, right=.98)


fig.supxlabel("Number of shared data sets")
fig.supylabel("Test log-likelihood")

if center == "all":
    fig.suptitle(f"More parties sharing improves analysis\n$\epsilon = {epsilon}$") #\n Test log-likelihoods for starting from all centers, $\epsilon = {epsilon}$
else:
    fig.suptitle(f"More parties sharing improves analysis\n Local center is {center}")

fig.savefig(os.path.join(args.output_path, f"{args.test_type}_over_num_shared_{center}_eps{epsilon}_max{int(args.max_center_size) if args.max_center_size >= 1 else args.max_center_size}_mc_{args.plot_variant}.pdf"), format="pdf")
plt.close()
p_vals.to_csv(os.path.join(args.output_path, f"{args.test_type}_over_num_shared_{center}_eps{epsilon}_max{int(args.max_center_size) if args.max_center_size >= 1 else args.max_center_size}_mc.csv"), header=True)

