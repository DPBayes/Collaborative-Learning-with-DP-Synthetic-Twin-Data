import jax, pickle, argparse, os

import numpy as np
import pandas as pd
from collections import defaultdict

from twinify.model_loading import load_custom_numpyro_model

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import filenamer, load_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("stored_model_dir", type=str, help="Dir from which to read learned parameters.")
    parser.add_argument("--output_dir", type=str, default=None, help="Dir to store the results")
    parser.add_argument("--epsilon", type=str, help="Privacy level")
    def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
    parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_generate", action="store_true", default=False, help="Use the batch generative mode: generate data sets from single draws of the parameter posterior")
    parser.add_argument("--num_synthetic_data_sets", "--M", default=100, type=int, help="Number of synthetic data sets to generate; only used when --batch_generate is set.")
    parser.add_argument("--mode", default="centers", type=str, help="Whether to run 'centers' or 'wholepop'")
    parser.add_argument("--convergence_test_threshold", default=0.05, type=float)
    parser.add_argument("--convergence_test_spacing", default=100, type=int)
    parser.add_argument("--max_center_size", default=None, type=float, help="The maximum size of data points to consider from each center.")

    args, unknown_args = parser.parse_known_args()

    if args.output_dir is None:
        args.output_dir = args.stored_model_dir

    ########################################## Read original data
    # read the whole UKB data
    df_whole = pd.read_csv(args.data_path)
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))

    train_df_whole = df_whole.copy()
    train_df_whole = train_df_whole.dropna()

    model, _, preprocess_fn, postprocess_fn= load_custom_numpyro_model(args.model_path, args, unknown_args, train_df_whole)

    train_data_whole, num_data_whole, feature_names = preprocess_fn(train_df_whole)

    conditioned_postprocess_fn = lambda posterior_samples: postprocess_fn(
        posterior_samples, df_whole, feature_names
    )
    ########################################## Set up guide

    from smart_auto_guide import SmartAutoGuide

    from numpyro.infer.autoguide import AutoDiagonalNormal

    observation_sites = {'X', 'y'}
    guide = SmartAutoGuide.wrap_for_sampling(AutoDiagonalNormal, observation_sites)(model)
    guide.initialize()

    ########################################## Prepare reusable sampling function
    ## NOTE this way of sampling seemed  is fast if always sampling the same number of synthetic draws!! I guess this
    ## prevents triggering recompilation in jax...
    ## We can still use it by just sampling more than we need and throw away the 'excess' samples
    from numpyro.infer import Predictive
    @jax.jit
    def jitted_ppd_sampler(model_params, posterior_rng, sample_rng):

        # sample single parameter vector
        posterior_sampler = Predictive(
            guide, params=model_params, num_samples=1
        )

        posterior_samples = posterior_sampler(posterior_rng)
        # models always add a superfluous batch dimensions, squeeze it
        posterior_samples = {k: v.squeeze(0) for k,v in posterior_samples.items()}

        # sample num_record_samples_per_parameter_sample data samples
        ppd_sampler = Predictive(model, posterior_samples, batch_ndims=0)

        ppd_sample = ppd_sampler(sample_rng)
        return ppd_sample

    def sample_synthetic(rng, num_synthetic, model_params):
        posterior_rng, ppd_rng = jax.random.split(rng)
        #
        per_sample_posterior_rngs = jax.random.split(posterior_rng, num_synthetic)
        per_sample_rngs = jax.random.split(ppd_rng, num_synthetic)

        if args.batch_generate:
            fixed_sampler = lambda sample_rng: jitted_ppd_sampler(model_params, posterior_rng, sample_rng)
            posterior_samples = jax.vmap(fixed_sampler)(per_sample_rngs)
        else:
            fixed_sampler = lambda sample_rng, posterior_rng: jitted_ppd_sampler(model_params, posterior_rng, sample_rng)
            posterior_samples = jax.vmap(fixed_sampler)(per_sample_rngs, per_sample_posterior_rngs)

        posterior_samples['X'] = np.clip(
            posterior_samples['X'], 0, np.max(train_data_whole[0], axis=0)
        )
        # models always add a superfluous batch dimensions, squeeze it
        squeezed_posterior_samples = {k: v.squeeze(1) for k, v in posterior_samples.items()}
        return conditioned_postprocess_fn(squeezed_posterior_samples)[1]

    ##########################################

    if args.mode != 'centers':
        raise Exception("main script can only do 'centers' mode from now on, but you specified a different mode")

    all_centers = list(df_whole['assessment_center'].unique())


    num_synthetic_max = train_df_whole["assessment_center"].value_counts().max()
    # if args.max_center_size is not None and num_synthetic_max > args.max_center_size:
    #     num_synthetic_max = args.max_center_size
    print(f"{num_synthetic_max=}")

    if args.batch_generate:
        sampling_rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.num_synthetic_data_sets)
    else:
        sampling_rngs = [jax.random.PRNGKey(args.seed)]

    synthetic_datasets = defaultdict(list)
    for sampling_idx, sampling_rng in enumerate(sampling_rngs):
        print(f"#### SAMPLING SYNTHETIC DATA SET {sampling_idx+1} / {args.num_synthetic_data_sets}")
        per_center_rngs = jax.random.split(sampling_rng, len(all_centers))
        for center, per_center_rng in zip(all_centers, per_center_rngs):
            print(f"  Sampling {center}")

            # read posterior params from file
            model_params = load_params(None, center, args)

            # sample synthetic data
            num_records_for_center = (train_df_whole.assessment_center == center).sum()

            encoded_syn_df = sample_synthetic(per_center_rng, num_synthetic_max, model_params).iloc[:num_records_for_center]
            print(f"Sampled data contains {encoded_syn_df.shape[0]} entries")

            synthetic_datasets[center].append(encoded_syn_df)

    ## store synthetic data for all centers
    twin_data_output_name = filenamer("synthetic_data", "all", args)
    twin_data_output_path = f"{os.path.join(args.output_dir, twin_data_output_name)}.p"
    with open(twin_data_output_path, "wb") as f:
        pickle.dump(synthetic_datasets, f)

if __name__ == "__main__":
    main()
