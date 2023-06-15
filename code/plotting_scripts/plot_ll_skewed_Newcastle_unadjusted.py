"""
Plots results comparing using only local center train data (adjusted and unadjusted)
vs adding synthetic data from other centers, evaluated as log-lik on global test data
for the very ethnically unbalanced Newcastle assessment center.

Used to create Fig 4.
"""
import os, sys

import numpy as np
import jax
import pandas as pd

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
parser.add_argument("--center", default="Newcastle")
parser.add_argument("--test_type", choices=["ll", "avg_ll"], default="avg_ll")
parser.add_argument("--mc_seed", default=9872635, type=int)
parser.add_argument("--num_mc_samples", default=100, type=int)

args = parser.parse_args()
args.max_center_size = None

##### fit non-dp model
# read original data
orig_data_train = pd.read_csv(args.train_data_path)
orig_data_test = pd.read_csv(args.test_data_path)


normalization_factor = 1.
if args.test_type == "avg_ll":
    normalization_factor = 1. / len(orig_data_test)

# population level fit
from utils import fit_model1, pred_test_likelihood, pred_test_likelihood_posterior_mc_approx

orig_data_train = pd.read_csv(args.train_data_path, index_col=0)
del orig_data_train["covid_test_present"]
del orig_data_train["covid_test_positive_in_hospital"]

# population level fit
orig_pop_level_fit = fit_model1(orig_data_train)[0]
pred_ll_orig_wholepop = pred_test_likelihood(orig_pop_level_fit.params, orig_data_test, return_average=False) * normalization_factor


######################
## Read combined fits
epsilon = args.epsilon
n_epochs = args.num_epochs

read_for_first_time = True

combined_pooled_coefs_for_seeds = dict()
local_only_per_center_coefs_for_seeds = dict()


def lls_to_df(lls) -> pd.DataFrame:
    num_seeds, num_repetitions, num_mc = lls.shape

    # seeds, reps, samples
    lls_flat = np.ravel(lls) * normalization_factor
    lls_seeds = np.repeat(np.arange(num_seeds), num_repetitions * num_mc)
    lls_reps = np.tile(np.repeat(np.arange(num_repetitions), num_mc), num_seeds)

    lls_df = pd.DataFrame({"lls": lls_flat, "seed": lls_seeds, "rep": lls_reps})

    return lls_df

for seed in range(123, 123+10):
    print(f"Processing seed {seed}")

    fname = filenamer("lls_over_num_shared", args.center, args, seed=None) + "_mc.p"
    fpath = os.path.join(args.lls_path, fname)

    if os.path.exists(fpath):
        lls = pd.read_pickle(fpath)
        assert len(lls.shape) == 4
        # seeds, reps, centers, samples
        local_only_lls = lls_to_df(lls[:, :, 0]).assign(variant='local only')
        combined_lls = lls_to_df(lls[:, :, -1]).assign(variant='combined')
        lls_df = pd.concat((local_only_lls, combined_lls))

    else:
        print(f"Missing seed {seed}")
        print(f"looked for {fpath}")


# compute fit for center without adjusting for ethnicity
orig_data_train_center = orig_data_train[orig_data_train.assessment_center == args.center]
center_unadjusted_fit = fit_model1(orig_data_train_center, include_ethnicity=False)[0]
num_seeds = len(lls_df['seed'].unique())
rng_key = jax.random.PRNGKey(args.mc_seed)
local_only_unadjusted_lls = pred_test_likelihood_posterior_mc_approx(
    orig_data_test, rng_key, statsmodels_result=center_unadjusted_fit, num_mc_samples=num_seeds * args.num_mc_samples, return_average=False
)
local_only_unadjusted_lls = local_only_unadjusted_lls.reshape(num_seeds, 1, args.num_mc_samples)
local_only_unadjusted_lls = lls_to_df(local_only_unadjusted_lls).assign(variant='local only, w/o ethnicity')
lls_df = pd.concat((lls_df, local_only_unadjusted_lls))

import significance_tests

combined_df = lls_df[lls_df.variant == "combined"]
local_only_df = lls_df[lls_df.variant == "local only"].groupby("seed").apply(lambda df: df[df.rep == 0])
local_only_una_df = lls_df[lls_df.variant == "local only, w/o ethnicity"]

p_vals = pd.Series([
    significance_tests.compute_ranked_t_test_p(combined_df, local_only_df, alternative="two-sided"),
    significance_tests.compute_ranked_t_test_p(combined_df, local_only_una_df, alternative="two-sided"),
    significance_tests.compute_ranked_t_test_p(local_only_una_df, local_only_df, alternative="two-sided")
], index=['combined_local', 'combined_local_unadjusted', 'local_unadjusted_local'])

print("#### p values for ranked Welch t-test ####")
print(p_vals)

import matplotlib.pyplot as plt
import seaborn as sns

from fig_style import get_fig_style

plt.rcParams.update(get_fig_style())

fig, axis = plt.subplots()
axis.axhline(pred_ll_orig_wholepop, ls="--", color="k")
sns.boxplot(data=lls_df, y='lls', x='variant', order=['combined', 'local only', 'local only, w/o ethnicity'], ax=axis, showfliers=False, linewidth=.5, width=.2)
axis.grid(axis='x', ls="-.", lw=".1", c="grey", alpha=.5)

axis.set_ylabel("Test log-likelihood")
axis.set_xlabel("")
axis.set_title(f"{args.center} center suffers from distribution skew\n$\epsilon={args.epsilon}$")

max_size_suffix = ""
if args.max_center_size is not None:
    max_size_suffix = f"_max{args.max_center_size}"
fig.subplots_adjust(left=.15, right=.98)
fig.savefig(f"{args.test_type}unadjusted_big_center_loglikelihood_eps{epsilon}_ne{n_epochs}_80_20_split{max_size_suffix}.pdf", format="pdf")
plt.close()
