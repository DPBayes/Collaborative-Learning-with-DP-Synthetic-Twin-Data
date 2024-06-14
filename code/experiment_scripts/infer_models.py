import jax, d3p, pickle, argparse, tqdm, os

import numpy as np
import pandas as pd

from numpyro.optim import Adam

from d3p.minibatch import q_to_batch_size, poisson_batchify_data, batch_size_to_q
from d3p.dputil import approximate_sigma_remove_relation

from twinify.model_loading import load_custom_numpyro_model
from twinify.infer import InferenceException
from twinify.results import TwinifyRunResult

import d3p.random
import chacha.defs
import secrets

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import filenamer

from collections import namedtuple
traces = namedtuple('traces', ['loc_trace', 'scale_trace'])

from dpsvi import AlignedGradientDPSVI

from twinify import __version__


def initialize_rngs(seed):
    if seed is None:
        seed = secrets.randbits(chacha.defs.ChaChaKeySizeInBits)
    master_rng = d3p.random.PRNGKey(seed)
    print(f"RNG seed: {seed}")

    inference_rng, sampling_rng, _ = d3p.random.split(master_rng, 3)
    sampling_rng = d3p.random.convert_to_jax_rng_key(sampling_rng)

    numpy_random_state = np.random.RandomState(seed)

    return inference_rng, sampling_rng, numpy_random_state


class InferenceException(Exception):

    def __init__(self, trace_tuple: traces):
        self.traces = trace_tuple

    @property
    def num_epochs(self):
        return len(self.traces.loc_trace)


def _train_model(rng, rng_suite, svi, data, batch_size, num_data, num_epochs, silent=False):
    rng, svi_rng, init_batch_rng = rng_suite.split(rng, 3)

    assert(type(data) == tuple)
    from twinify.infer import _cast_data_tuple
    data = _cast_data_tuple(data)
    q = batch_size_to_q(batch_size, num_data)
    init_batching, get_batch = poisson_batchify_data(data, q, .99, rng_suite=rng_suite)
    _, batchify_state = init_batching(init_batch_rng)

    batch, _ = get_batch(0, batchify_state)
    svi_state = svi.init(svi_rng, *batch)

    @jax.jit
    def train_epoch(num_iters_for_epoch, svi_state, batchify_state):
        def update_iteration(i, state_and_loss):
            svi_state, loss = state_and_loss
            batch, mask = get_batch(i, batchify_state)
            svi_state, iter_loss = svi.update(svi_state, *batch, mask=mask)
            return (svi_state, loss + iter_loss / num_iters_for_epoch)

        return jax.lax.fori_loop(0, num_iters_for_epoch, update_iteration, (svi_state, 0.))

    rng, epochs_rng = rng_suite.split(rng)

    progressbar = tqdm.tqdm(range(num_epochs))
    for e in progressbar:
        batchify_rng = rng_suite.fold_in(epochs_rng, e)
        num_batches, batchify_state = init_batching(batchify_rng)

        svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
        params_after_epoch = svi.get_params(svi_state)
        if e == 0:
            locs_over_epochs = np.zeros((num_epochs, params_after_epoch['auto_loc'].shape[-1]))
            scales_over_epochs = np.zeros((num_epochs, params_after_epoch['auto_scale'].shape[-1]))
        locs_over_epochs[e] = params_after_epoch['auto_loc']
        scales_over_epochs[e] = params_after_epoch['auto_scale']
        if np.isnan(loss):
            raise InferenceException(
                traces(locs_over_epochs, scales_over_epochs)
            )
        loss /= num_data
        progressbar.set_description(f"epoch {e}: loss {loss}")

    return svi.get_params(svi_state), loss, \
        traces(locs_over_epochs, scales_over_epochs)

def run_inference(args, train_data, model, guide):

    num_data = len(train_data[0])

    # split rngs
    inference_rng, sampling_rng, numpy_random_state = initialize_rngs(args.seed)

    if args.epsilon != 'non_dp':
        delta = 1e-6
        num_total_iters = np.ceil(args.num_epochs / args.sampling_ratio)
        dp_scale, _, _ = approximate_sigma_remove_relation(
                float(args.epsilon),
                delta,
                args.sampling_ratio,
                num_total_iters
        )
    else:
        dp_scale = 0.0
    print(f"Adding noise with std. {dp_scale}")

    # set up dpsvi algorithm of choice
    optimizer = Adam(1e-3)

    svi = AlignedGradientDPSVI(
        model,
        guide,
        optimizer,
        args.clipping_threshold,
        dp_scale,
        num_obs_total=num_data
    )

    # train the model
    batch_size = q_to_batch_size(args.sampling_ratio, num_data)
    posterior_params, elbo, _ = _train_model(
            inference_rng, d3p.random,
            svi, train_data,
            batch_size, num_data, args.num_epochs
    )

    return posterior_params, elbo

def store_results(name, posterior_params, elbo, args, unknown_args):
    # save results
    result = TwinifyRunResult(
        posterior_params, elbo, args, unknown_args, __version__
    )

    output_name = filenamer(None, name, args)
    output_path = f"{os.path.join(args.output_dir, output_name)}"
    print(f"Storing results to {output_path}")
    pickle.dump(result, open(f"{output_path}.p", "wb"))


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("output_dir", type=str, help="Dir to store outputs.")
    parser.add_argument("--epsilon", type=str, help="Privacy level")
    def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
    parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
    parser.add_argument("--mode", default="centers", type=str, help="Whether to run 'centers' or 'wholepop'")
    parser.add_argument("--max_center_size", default=None, type=float, help="The maximum size of data points to consider from each center. Either an absolute value or a fraction.")

    args, unknown_args = parser.parse_known_args()

    # read the whole UKB data
    try:
        df_whole = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        return -1
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))

    train_df_whole = df_whole.copy()
    train_df_whole = train_df_whole.dropna()

    try:
        model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path, args, unknown_args, train_df_whole)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        print("#### COULD NOT FIND THE MODEL FILE ####")
        print(e)
        return -1

    train_data_whole, num_data_whole, feature_names = preprocess_fn(train_df_whole)

    #
    assert isinstance(train_data_whole, tuple)
    if len(train_data_whole) == 1:
        print("After preprocessing, the data has {} entries with {} features each.".format(*train_data_whole[0].shape))
    else:
        print("After preprocessing, the data was split into {} splits:".format(len(train_data_whole)))
        for i, x in enumerate(train_data_whole):
            print("\tSplit {} has {} entries with {} features each.".format(i, x.shape[0], 1 if x.ndim == 1 else x.shape[1]))

    if args.mode != 'centers':
        print("Processing all data")
        train_data = train_data_whole
        print("Data contains {} entries with {} dimensions".format(*train_data[0].shape))

        posterior_params, elbo = run_inference(args, train_data, model, guide)
        store_results("wholepop", posterior_params, elbo, args, unknown_args)

    else:
        ## We read the centers and learn the generative model separately for each center
        ## Why this way and not just calling this code separately on each center? Well, in the current form
        ## the data encoding is based on the number of categories present in the data. BUT if we apply the
        ## encoding (here preprocessing) to the whole data, and after that split the centers, we will have
        ## the same encoding for each of the centers which then makes the comparison much more robust.

        all_centers = list(df_whole["assessment_center"].unique())

        for center in all_centers:
            print(f"Processing {center}")
            # compute the needed dp scale for the dp-sgd
            train_data = tuple(elem[train_df_whole["assessment_center"] == center] for elem in train_data_whole)
            print("Data for center contains {} entries with {} dimensions".format(*train_data[0].shape))

            posterior_params, elbo = run_inference(args, train_data, model, guide)
            store_results(center, posterior_params, elbo, args, unknown_args)


if __name__ == "__main__":
    main()
