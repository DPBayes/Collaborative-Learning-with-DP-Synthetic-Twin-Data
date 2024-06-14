"""
Computes the evolution of downstream performance (measured in log-likelihood)
at a single center over the number of available synthetic data sets from other centers
based on sampled synthetic data from generate_twin_data.py.
"""
import os

import numpy as np
import pandas as pd

import typing
import jax
from joblib import Parallel, delayed

import argparse

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import filenamer, fit_model1, pred_test_likelihood_posterior_mc_approx
from pvalue_utils import pool_analysis

import tqdm
from collections import namedtuple
DownstreamFit = namedtuple("DownstreamFit", ("params", "bse"))

parser = argparse.ArgumentParser()
parser.add_argument("center", type=str, help="The center for which to plot the improvement")
parser.add_argument("--train_data_path", default=None)
parser.add_argument("--test_data_path", default=None)
parser.add_argument("--syn_data_path", default=None)
parser.add_argument("--output_path", default=None)
parser.add_argument("--epsilon", default=1.0, type=str, help="Privacy level")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=2.0, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--max_center_size", default=None, type=float, help="The maximum size of data points to consider from each center.")
parser.add_argument("--filter_extreme_lls", default=0., type=float, help="The cutoff percentage of extreme LLs do discard before plotting (i.e., a value of .2 will only plot mean and std over the 80 perc. confidence interval (.1 -> .9).")
parser.add_argument("--permutation_seed", default=681234, type=int, help="RNG seeds for permuting the list of centers.")
parser.add_argument("--num_repetitions", default=100, type=int, help="How many permutations to sample for creating the plot.")
parser.add_argument("--mc_seed", default=6782352, type=int)
parser.add_argument("--num_mc_samples", default=1000, type=int)
parser.add_argument("--num_processes", default=1, type=int)
parser.add_argument("--cont", default=False, action='store_true', help="If set, tries to continue a previously interrupted run")

args = parser.parse_args()
print(args)

mc_base_key = jax.random.PRNGKey(args.mc_seed)

##### fit non-dp model
# read original data
orig_data_train = pd.read_csv(args.train_data_path, index_col=0)
del orig_data_train["covid_test_present"]
del orig_data_train["covid_test_positive_in_hospital"]
orig_data_test = pd.read_csv(args.test_data_path, index_col=0)

# population level fit

center = args.center

epsilon = args.epsilon
n_epochs = args.num_epochs

lls_for_all_seeds = []

fname_output = filenamer("lls_over_num_shared", center, args) + "_mc.p"
fpath_output = os.path.join(args.output_path, fname_output)
if args.cont:
    if os.path.exists(fpath_output):
        lls_for_all_seeds_arr = pd.read_pickle(fpath_output)
        lls_for_all_seeds = list(lls_for_all_seeds_arr)

seeds = range(123 + len(lls_for_all_seeds), 123 + 10)
for seed in tqdm.tqdm(seeds, "seed"):
    mc_seed_key = jax.random.fold_in(mc_base_key, seed)

    twin_data_name = filenamer("synthetic_data", "all", args, seed=seed)
    twin_data_path = os.path.join(args.syn_data_path, twin_data_name + ".p")

    test_data = orig_data_test

    local_only_df = orig_data_train[orig_data_train["assessment_center"] == center]
    if os.path.exists(twin_data_path):
        twin_data_sets = pd.read_pickle(twin_data_path)  # type: typing.Dict[str, typing.List[pd.DataFrame]]
    else:
        print(f"Missing twin data sets for seed {seed}")
        print(f"loooked for {twin_data_path}")
        continue

    centers = list(twin_data_sets.keys())
    other_centers = sorted(list(set(centers) - set((center,))))

    mc_seed_local_only_key, mc_seed_key = jax.random.split(mc_seed_key, 2)
    local_only_fit, _, reference_groups = fit_model1(local_only_df)
    local_only_lls = pred_test_likelihood_posterior_mc_approx(
        test_data, mc_seed_local_only_key, statsmodels_result=local_only_fit, return_average=False, num_mc_samples=args.num_mc_samples
    )

    def fit_downstream_model(df):
        try:
            res = fit_model1(df, reference_groups=reference_groups)[0]
            return DownstreamFit(res.params, res.bse)
        except Exception:
            return DownstreamFit(local_only_fit.params * np.nan, local_only_fit.bse * np.nan)

    num_syn_data_sets = len(twin_data_sets[other_centers[0]])

    lls_for_seed = np.zeros((args.num_repetitions, len(other_centers) + 1, args.num_mc_samples))
    lls_for_seed[:, 0] = np.array(local_only_lls)

    permutation_rs = np.random.RandomState(args.permutation_seed) # reset random state so that permutations are consistent over all seeds
    def run_single_permutation(rep, other_centers_in_joining_order, use_tqdm=False):
        mc_rep_key = jax.random.fold_in(mc_seed_key, rep)

        combined_dfs = [local_only_df.copy() for i in range(num_syn_data_sets)]

        lls_for_permutation = np.zeros((len(other_centers_in_joining_order), args.num_mc_samples))

        iterable = other_centers_in_joining_order
        if use_tqdm:
            iterable = tqdm.tqdm(iterable, "centers", leave=False)

        for j, next_center in enumerate(iterable):
            mc_next_key = jax.random.fold_in(mc_rep_key, j)
            combined_dfs = [
                pd.concat((combined_df, twin_data_sets[next_center][i].assign(assessment_center=next_center)))
                for i, combined_df in enumerate(combined_dfs)
            ]

            combined_model_fits = [
                fit_downstream_model(combined_df) for combined_df in combined_dfs
            ]

            coefs = pd.concat([fit.params for fit in combined_model_fits], axis=1)
            stderrs = pd.concat([fit.bse for fit in combined_model_fits], axis=1)

            pooled_coefs, pooled_vars = pool_analysis(coefs, stderrs**2, return_pvalues=False)

            #with_next_center_pred_ll = pred_test_likelihood(pooled_coefs, test_data)
            with_next_center_pred_lls = pred_test_likelihood_posterior_mc_approx(
                test_data, mc_next_key, coef_mean=pooled_coefs, coef_vars=pooled_vars, return_average=False, num_mc_samples=args.num_mc_samples
            )

            lls_for_permutation[j] = np.array(with_next_center_pred_lls)
        return lls_for_permutation

    permutations = list(permutation_rs.permutation(other_centers) for _ in range(args.num_repetitions))

    lls_for_permutations = Parallel(n_jobs=args.num_processes, batch_size=1)(
        delayed(run_single_permutation)(rep, permutation)
        for rep, permutation in enumerate(tqdm.tqdm(permutations, "Repetitions", leave=False))
    )

    lls_for_permutations = np.array(lls_for_permutations)
    lls_for_seed[:, 1:] = lls_for_permutations
    lls_for_all_seeds.append(lls_for_seed)

    lls_for_all_seeds_arr = np.array(lls_for_all_seeds)
    pd.to_pickle(lls_for_all_seeds_arr, fpath_output)
