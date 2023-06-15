"""
Plots results comparing using only local center train data vs adding synthetic data
from all other centers, evaluated as log-lik on global test data.

Used to create Fig 1.
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
parser.add_argument("--train_data_path", default=None)
parser.add_argument("--test_data_path", default=None)
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
parser.add_argument("--plot_without_barts", default=False, action='store_true')
parser.add_argument("--annotate_significance", default=False, action='store_true')

args = parser.parse_args()

##### fit non-dp model
# read original data
orig_data_train = pd.read_csv(args.train_data_path)
orig_data_test = pd.read_csv(args.test_data_path)

# population level fit
from utils import fit_model1, pred_test_likelihood_posterior_mc_approx
orig_pop_level_fit = fit_model1(orig_data_train)[0]


centers = list(orig_data_train["assessment_center"].unique())

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

def test_fn(test_data, rng_key, **kwargs):
    return pred_test_likelihood_posterior_mc_approx(test_data, rng_key, return_average=False, **kwargs)

pred_ll_orig_wholepop = test_fn(orig_data_test, orig_ll_rng_key, statsmodels_result=orig_pop_level_fit) * normalization_factor

for i, center in enumerate(centers):

    print(f"Processing center {center}")

    fname_lls = filenamer("lls_over_num_shared", center, args) + "_mc.p"
    fpath_lls = os.path.join(args.lls_path, fname_lls)
    lls = pd.read_pickle(fpath_lls)
    # lls.shape == (seed, permutation_repeat, position, lls samples)

    for i in range(10):
        seed = 123 + i
        pred_ll_local_only_dict[center][seed] = lls[i, 0, 0].ravel() * normalization_factor # samples for first position are copies of the local-only samples across all 'rep'; this will skew tests, so we just use the first

        pred_ll_combined_dict[center][seed] = lls[i, :, -1].ravel() * normalization_factor

## plot

# def mean_without_extreme(x: pd.Series, alpha: float=args.filter_extreme_lls):
#     if alpha > 0:
#         quantiles = x.quantile((alpha/2, 1. - alpha/2))
#         filtered = x[(x < quantiles.iloc[1]) & (x > quantiles.iloc[0])]
#     else:
#         filtered = x
#     return pd.Series({'mean': filtered.mean(), 'stderr': filtered.std(ddof=1) / np.sqrt(len(filtered))}, name=x.name)

# sort centers based on the likelihood evaluated by the per-center approach

pred_ll_local_only_df = pd.concat([
    pd.DataFrame(pred_ll_local_only_dict[center]).melt(value_name="lls", var_name="seed").assign(center=center) for center in pred_ll_local_only_dict
])
sorted_centers = list(pred_ll_local_only_df.groupby("center").median()['lls'].sort_values().index)
center_labels = sorted_centers

pred_ll_combined_df = pd.concat([
    pd.DataFrame(pred_ll_combined_dict[center]).melt(value_name="lls", var_name="seed").assign(center=center) for center in pred_ll_combined_dict
])

plot_df = pd.concat((
    pred_ll_combined_df.assign(variant="combined"),
    pred_ll_local_only_df.assign(variant="local only"),
))

import significance_tests
def compute_ranked_t_test_p(center_df):
    combined_df = center_df[center_df.variant == "combined"]
    local_only_df = center_df[center_df.variant == "local only"]
    return significance_tests.compute_ranked_t_test_p(combined_df, local_only_df)
    
p_vals = plot_df.groupby("center").apply(compute_ranked_t_test_p)
if args.annotate_significance:
    center_labels = significance_tests.annotate_significance(p_vals, sorted_centers)

print("#### p values for ranked Welch t-test ####")
print(p_vals)

import matplotlib.pyplot as plt
import seaborn as sns

from fig_style import get_fig_style
plt.rcParams.update(get_fig_style(aspect_ratio=12/12))

plot_regions = [(-1.6, -.5), (-4.3, -3.1)]
plot_region_heights = [reg[1] - reg[0] for reg in plot_regions]

def plot(axis):
    sns.boxplot(data=plot_df, x='center', y='lls', hue='variant', order=sorted_centers, ax=axis, showfliers=False, linewidth=.5, width=.5)
    axis.axhline(pred_ll_orig_wholepop.mean(), ls="--", lw=1, color="k", label="full population")
    axis.grid(axis='x', ls="-.", lw=".1", c="grey", alpha=.5)

if args.plot_without_barts:
    plt.rcParams.update(get_fig_style(aspect_ratio=12/8))
    fig, ax1 = plt.subplots(1, 1)

    plot(ax1)
    ax1.set_ylim(plot_regions[0])
    lowest_ax = ax1

else:
    plt.rcParams.update(get_fig_style(aspect_ratio=12/12))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': plot_region_heights})

    plot(ax1)
    ax1.set_ylim(plot_regions[0])
    plot(ax2)
    ax2.set_ylim(plot_regions[1])

    # hide the spines between ax and ax2
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.get_legend().remove()

    ax1.set_xlabel(None)
    ax1.set_ylabel(None)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.subplots_adjust(hspace=.05)
    lowest_ax = ax2

fig.suptitle(f"Analysis performance using synthetic twins\n$\epsilon = {epsilon}$")
fig.supylabel("Test log-likelihood")
lowest_ax.set_xticks(np.arange(len(sorted_centers)))
lowest_ax.set_xticklabels(center_labels, rotation=45, ha='right', va='top')
lowest_ax.set_ylabel(None)
lowest_ax.set_xlabel("Center")
lowest_ax.legend()

max_size_suffix = ""
if args.max_center_size is not None:
    max_size_suffix = f"_max{args.max_center_size}"
plt.savefig(f"{args.test_type}_combined_test_loglikelihood_eps{epsilon}_ne{n_epochs}_80_20_split{max_size_suffix}{'_withoutBarts' if args.plot_without_barts else ''}.pdf", format="pdf", bbox_inches="tight")
plt.close()
