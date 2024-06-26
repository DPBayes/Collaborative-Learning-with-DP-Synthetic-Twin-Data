from numpyro import sample, plate
import numpyro.distributions as dists
from numpyro.infer.autoguide import AutoDiagonalNormal
import jax.numpy as jnp
import numpy as np
import pandas as pd
import argparse
from typing import Iterable
from collections import OrderedDict
from functools import partial

import sys
sys.path.append("./twinify_models")
from mixture_model import MixtureModel

reference_groups = OrderedDict(
    age_group="[40, 45)",
    sex="Female",
    ethnicity="White British",
    deprivation="1st",
    education="College or University degree",
    #assessment_center="Newcastle"
)

def preprocess(ori_df):
    df = pd.DataFrame()
    df['age_group'] = ori_df['age_group'].astype('category').cat.codes
    df['sex'] = ori_df['sex'].astype('category').cat.codes
    df['ethnicity'] = ori_df['ethnicity'].astype('category').cat.codes
    df['deprivation'] = ori_df['deprivation'].astype('category').cat.codes
    df['education'] = ori_df['education'].astype('category').cat.codes
    df['covid_test_result'] = ori_df['covid_test_result'].astype(bool).map({False: 0, True: 1})
    y = df.pop('covid_test_result')
    X = df
    num_data = len(X)
    return (X, y), num_data

def postprocess(posterior_samples, ori_df):
    syn_X = posterior_samples['X']
    syn_y = posterior_samples['y']
    syn_data = jnp.hstack([syn_X, syn_y[:, np.newaxis]])
    syn_df = pd.DataFrame(np.array(syn_data), columns = [
        'age_group', 'sex', 'ethnicity', 'deprivation', 'education', 'covid_test_result'
    ])

    age_group_labels = ori_df['age_group'].astype('category').cat.categories
    sex_labels = ori_df['sex'].astype('category').cat.categories
    ethnicity_labels = ori_df['ethnicity'].astype('category').cat.categories
    deprivation_labels = ori_df['deprivation'].astype('category').cat.categories
    education_labels = ori_df['education'].astype('category').cat.categories

    encoded_syn_df = syn_df.copy()
    encoded_syn_df['age_group'] = pd.Categorical.from_codes(encoded_syn_df['age_group'], age_group_labels)
    encoded_syn_df['sex'] = pd.Categorical.from_codes(encoded_syn_df['sex'], sex_labels)
    encoded_syn_df['ethnicity'] = pd.Categorical.from_codes(encoded_syn_df['ethnicity'], ethnicity_labels)
    encoded_syn_df['deprivation'] = pd.Categorical.from_codes(encoded_syn_df['deprivation'], deprivation_labels)
    encoded_syn_df['education'] = pd.Categorical.from_codes(encoded_syn_df['education'], education_labels)
    encoded_syn_df['covid_test_result'] = encoded_syn_df['covid_test_result'].astype(bool)
    return syn_df, encoded_syn_df


def onehot_encode(x, num_values):
    lut = jnp.eye(num_values)
    return lut[x]

def adjusted_onehot_encode(x, num_values, reference_idx):
    encoded = onehot_encode(x, num_values)
    return jnp.hstack((encoded[:, :reference_idx], encoded[:, (reference_idx+1):]))

def model_factory(twinify_args: argparse.Namespace, unparsed_args: Iterable[str], ori_df: pd.DataFrame):
    model_args_parser = argparse.ArgumentParser("Model 1")
    model_args_parser.add_argument('--scale_cat_priors', action='store_true', default=False, help='Normalise Dirichlet priors by dividing with number of categories.')
    model_args_parser.add_argument('--scale_mix_priors', action='store_true', default=False, help='Normalise Dirichlet priors for mixture weights by dividing with number of categories.')
    model_args_parser.add_argument('--init_scale', type=float, default=0.1, help='Initial value for scales in variational AutoDiagonalNormal.')
    args = model_args_parser.parse_args(unparsed_args, twinify_args)

    # collect number of categories and index of reference group/category for each feature
    d_sizes = OrderedDict()
    reference_indices = OrderedDict()
    for feature, reference_category in reference_groups.items():
        idx = list(ori_df[feature].astype('category').cat.categories).index(reference_category)
        reference_indices[feature] = idx
        d_sizes[feature] = len(ori_df[feature].astype('category').cat.categories)

    prior_scale_age_group = 1.
    prior_scale_ethnicities = 1.
    prior_scale_sex = 1.
    prior_scale_deprivation = 1.
    prior_scale_education = 1.
    prior_scale_pis = 1.

    if args.scale_cat_priors:
        prior_scale_age_group = 1./d_sizes['age_group']
        prior_scale_sex = 1./d_sizes['sex']
        prior_scale_ethnicities = 1./d_sizes['ethnicity']
        prior_scale_deprivation = 1./d_sizes['deprivation']
        prior_scale_education = 1./d_sizes['education']

    if args.scale_mix_priors:
        prior_scale_pis = 1./args.k


    def model(X=None, y=None, num_obs_total=None):
        d = 5
        batch_size = 1

        if y is not None:
            assert X is not None
            batch_size = y.shape[0]

        if X is not None:
            assert d == X.shape[1]
            if y is None:
                batch_size = X.shape[0]
            else:
                assert X.shape[0] == batch_size


        if num_obs_total is None:
            num_obs_total = batch_size

        ############## PRIORs ##################

        # Prior for mixture model of regressors
        k = args.k
        age_group_probs = sample('age_group_probs', dists.Dirichlet(jnp.ones((k, d_sizes['age_group']))*prior_scale_age_group))
        age_group_dist = dists.Categorical(probs=age_group_probs)

        sex_probs = sample('sex_probs', dists.Beta(jnp.ones((k,))*prior_scale_sex, jnp.ones((k,))*prior_scale_sex))
        sex_dist = dists.Bernoulli(probs=sex_probs)

        ethnicity_probs = sample('ethnicity_probs', dists.Dirichlet(jnp.ones((k, d_sizes['ethnicity']))*prior_scale_ethnicities))
        ethnicity_dist = dists.Categorical(probs=ethnicity_probs)

        deprivation_probs = sample('deprivation_probs', dists.Dirichlet(jnp.ones((k, d_sizes['deprivation']))*prior_scale_deprivation))
        deprivation_dist = dists.Categorical(probs=deprivation_probs)

        education_probs = sample('education_probs', dists.Dirichlet(jnp.ones((k, d_sizes['education']))*prior_scale_education))
        education_dist = dists.Categorical(probs=education_probs)

        pis = sample('pis_probs', dists.Dirichlet(jnp.ones(k)*prior_scale_pis))

        mixture_dist = MixtureModel(
            [age_group_dist, sex_dist, ethnicity_dist, deprivation_dist, education_dist],
            pis
        )

        # Set up Poisson regression: this sets the prior for the regression weights
        # and also prepares functions that convert each categorical into (adjusted) one-hot vectors
        d_onehot = 0
        poisson_coefs = []
        feature_encoders = []
        for feature, d_size in d_sizes.items():
            adjusted_d_size = d_size - 1
            d_onehot += adjusted_d_size

            feature_coefs = sample(f"covid_weight_{feature}", dists.Normal(jnp.zeros(adjusted_d_size), jnp.ones(adjusted_d_size)))
            poisson_coefs.append(feature_coefs)
            feature_encoders.append(partial(adjusted_onehot_encode, num_values=d_size, reference_idx=reference_indices[feature]))


        covid_test_weight = jnp.concatenate(poisson_coefs)
        assert covid_test_weight.shape == (d_onehot,)

        covid_test_intercept = sample("covid_intercept", dists.Normal(0., 1.))

        ############# LIKELIHOOD ####################

        with plate('batch', num_obs_total, batch_size):
            X = sample('X', mixture_dist, obs=X)

            encoded_X = jnp.hstack([
                encode(X[:, i]) for i, encode in enumerate(feature_encoders)
            ])
            assert(encoded_X.shape[1] == d_onehot)

            covid_test_logit = encoded_X @ covid_test_weight + covid_test_intercept
            y = sample('y', dists.Poisson(rate=jnp.exp(covid_test_logit)), obs=y)

    guide = AutoDiagonalNormal(model, init_scale=args.init_scale)

    return model, guide
