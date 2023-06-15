import jax, numpyro, argparse

import pandas as pd

import numpyro.handlers

from twinify.model_loading import load_custom_numpyro_model

"""
Computes log-likelihood MC samples using the model output from twinify directly, without synthetic data.
"""

from smart_auto_guide import SmartAutoGuide

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument('twinify_output_path', type=str, help='Path to twinify output.')
parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
parser.add_argument("output_dir", type=str, help="Dir to store outputs.")
parser.add_argument("--test_data_path", type=str)
parser.add_argument("--epsilon", type=str, help="Privacy level")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
parser.add_argument("--max_center_size", default=None, type=float, help="The maximum size of data points to consider from each center.")
parser.add_argument("--num_mc_samples", type=int, default=100)

args, unknown_args = parser.parse_known_args()

posterior_params = pd.read_pickle(args.twinify_output_path)[0]

# read the whole UKB data
try:
    test_df = pd.read_csv(args.test_data_path)
except Exception as e:
    print("#### UNABLE TO READ DATA FILE ####")
    print(e)

model, _, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path, args, unknown_args, test_df)

test_data, num_test_data, _ = preprocess_fn(test_df)
test_x, test_y = test_data[0].to_numpy(), test_data[1].to_numpy()

guide = SmartAutoGuide.wrap_for_sampling_and_initialize(numpyro.infer.autoguide.AutoDiagonalNormal, ["X", "y"], test_x)(model)

rng_key = jax.random.PRNGKey(args.seed)

param_samples = numpyro.infer.Predictive(guide, params=posterior_params, num_samples=args.num_mc_samples)(rng_key)

def get_loglikelihood(param_sample):
    tr = numpyro.handlers.trace(numpyro.handlers.substitute(model, param_sample)).get_trace(test_x, test_y, num_obs_total=num_test_data)
    return tr['y']['fn'].log_prob(tr['y']['value']).sum()

lls = jax.vmap(get_loglikelihood)(param_samples)
