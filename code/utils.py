import re
import numpy as np
import argparse

def filenamer(prefix, center, args=None, **kwargs):
    """ Helper function to derive filenames from experiment parameters. """
    def filenamer_explicit(prefix, center, epsilon=None, clipping_threshold=None, k=None, seed=None, num_epochs=None, num_iterations=None, max_center_size=None, **unused_kwargs):
        if prefix is None:
            prefix = ""
        max_size_suffix = ""
        if max_center_size is not None:
            max_size_suffix = f"_max{int(max_center_size) if max_center_size >= 1 else max_center_size}"
        if num_iterations is not None and num_epochs is None:
            num_epochs = num_iterations
        output_name=f"{prefix}_{epsilon}_C{clipping_threshold}_k{k}_seed{seed}_epochs{num_epochs}_{center}{max_size_suffix}"
        return output_name

    if isinstance(args, argparse.Namespace):
        new_kwargs = args.__dict__.copy()

        new_kwargs.update(kwargs)
        if 'prefix' in new_kwargs:
            del new_kwargs['prefix']
        if 'center' in new_kwargs:
            del new_kwargs['center']
        kwargs = new_kwargs

    return filenamer_explicit(prefix, center, **kwargs)


# STATSMODELS for downstream inference
import statsmodels.api as sm
import statsmodels.formula.api as smf
per_center_formula = "covid_test_result ~ "\
    "C(age_group, Treatment(reference='[40, 45)')) + " \
    "C(sex, Treatment(reference='Female')) + " \
    "C(ethnicity, Treatment(reference='White British')) + "\
    "C(deprivation, Treatment(reference='1st')) + " \
    "C(education, Treatment(reference='College or University degree'))"

wholepop_formula = per_center_formula + " +"\
    "C(assessment_center, Treatment(reference='Newcastle'))"

per_center_formula_no_ref_groups = "covid_test_result ~ "\
    "age_group + sex + ethnicity + deprivation + education"

per_center_formula_no_ref_groups = "covid_test_result ~ "\
    "age_group + sex + ethnicity + deprivation + education"

wholepop_formula_no_ref_groups = per_center_formula_no_ref_groups + " +"\
    "assessment_center"

standard_reference_groups = {
    'age_group': '[40, 45)',
    'sex': 'Female',
    'ethnicity': 'White British',
    'deprivation': '1st',
    'education': 'College or University degree',
    'assessment_center': 'Newcastle'
}

def make_model1_formula(
        reference_groups = dict(),
        include_age_group=True, include_sex=True, include_ethnicity=True,
        include_deprivation=True, include_education=True, include_center=False
    ):

    def make_adjustment_string(feature_name):
        if feature_name in reference_groups:
            return f"C({feature_name}, Treatment(reference='{reference_groups[feature_name]}'))"
        return feature_name

    adjustments = []
    if include_age_group: adjustments.append("age_group")
    if include_sex: adjustments.append("sex")
    if include_ethnicity: adjustments.append("ethnicity")
    if include_deprivation: adjustments.append("deprivation")
    if include_education: adjustments.append("education")
    if include_center: adjustments.append("assessment_center")

    return "covid_test_result ~ " + " + ".join(make_adjustment_string(adjustment) for adjustment in adjustments)


def fit_model1(data, reference_groups=None, set_standard_reference_groups=False, **kwargs):
    data = data.copy()

    if reference_groups is None:
        if set_standard_reference_groups:
            reference_groups = dict(standard_reference_groups)
        else:
            reference_groups = {
                feature_name: data[feature_name].value_counts().index[0]
                for feature_name in data
            }

    formula = make_model1_formula(reference_groups, **kwargs)

    data['covid_test_result'] = data['covid_test_result'].astype(int)
    model1 = smf.glm(formula=formula, family=sm.families.Poisson(), data=data)
    res = model1.fit(cov_type='HC1')
    return res, model1, reference_groups

def make_names_pretty(name):
    if "reference" in name:
        feature, group = re.search("C\((\w+).*T\.(.*)]", name).groups()
        return f"{feature}: {group}"
    elif "T." in name:
        feature, group = re.search("(\w+).*T\.(.*)]", name).groups()
        return f"{feature}: {group}"
    else:
        return name

from numpyro.distributions import Poisson, MultivariateNormal
from numpyro.diagnostics import split_gelman_rubin
import jax.numpy as jnp

def onehot_regressors(test_data, regressor_features):
    adjusted_onehotted_X_test = np.empty((len(test_data), 0))
    for feature in regressor_features:
        if feature == "Intercept":
            adjusted_onehotted_X_test = np.hstack((adjusted_onehotted_X_test, np.ones((len(test_data), 1))))
        else:
            feature, category = feature.split(":")
            category = category.strip()
            binary_feature = 1. * (test_data[feature].values == category)[:, np.newaxis]
            adjusted_onehotted_X_test = np.hstack((adjusted_onehotted_X_test, binary_feature))

    return adjusted_onehotted_X_test


def pred_test_likelihood(coef, test_data, return_average=True):
    """
    coef: regression coefficients, a pandas Series
    """
    factor_test_ll = 1.
    if return_average:
        factor_test_ll = 1. / len(test_data)

    test_y = test_data["covid_test_result"]

    # we need to choose test features based on what remained in the small center
    # for example, it is very likely that a certain minority was not included in the small centers
    renamed_existing_coefs = coef.dropna().rename(make_names_pretty)
    coefficients_from_fit = renamed_existing_coefs.index

    adjusted_onehotted_X_test = onehot_regressors(test_data, coefficients_from_fit)

    w = renamed_existing_coefs[coefficients_from_fit]
    rate = np.exp(adjusted_onehotted_X_test @ w)
    logl = Poisson(rate).log_prob(test_y.values).sum() * factor_test_ll
    return logl


from numpyro.primitives import sample, deterministic
from numpyro.infer import NUTS, MCMC, Predictive


def pred_test_likelihood_posterior_mc_approx(test_data, rng_key, *, statsmodels_result=None, coef_mean=None, coef_vars=None, alternative_test_data=None, num_mc_samples=1000, return_average=True, thinning_factor=1, use_mcmc=False):
    if statsmodels_result is not None:
        coef_mean = statsmodels_result.params
        coef_vars = statsmodels_result.bse**2

    if coef_mean is None:
        raise ValueError(f"Requires either coef_mean or statsmodels_result to not be None.")

    if coef_vars is None:
        raise ValueError(f"Requires either coef_vars or statsmodels_result to not be None.")

    # we need to choose test features based on what remained in the small center
    # for example, it is very likely that a certain minority was not included in the small centers
    existing_coef_idx = coef_mean.dropna().index
    renamed_existing_coefs = coef_mean.loc[existing_coef_idx].rename(make_names_pretty)
    renamed_existing_vars = coef_vars.loc[existing_coef_idx].rename(make_names_pretty).fillna(1e-3)#.fillna(???) TODO: sometimes these may be NaN if the coef is not; why? what to do?

    coefficients_from_fit = renamed_existing_coefs.index
    adjusted_onehotted_X_test = onehot_regressors(test_data, coefficients_from_fit)
    test_y = test_data["covid_test_result"].to_numpy()

    if alternative_test_data is not None:
        adjusted_onehotted_X_alt_test = onehot_regressors(alternative_test_data, coefficients_from_fit)
        alt_test_y = alternative_test_data["covid_test_result"].to_numpy()

    factor_test_ll = 1.
    factor_alt_test_ll = 1.
    if return_average:
        factor_test_ll = 1. / len(test_data)
        if alternative_test_data is not None:
            factor_alt_test_ll = 1. / len(alternative_test_data)

    def model():
        w = sample('w', MultivariateNormal(
                renamed_existing_coefs[coefficients_from_fit].to_numpy(),
                scale_tril=np.diag(np.sqrt(renamed_existing_vars[coefficients_from_fit].to_numpy()))
            )
        )
        rate = jnp.exp(adjusted_onehotted_X_test @ w)
        logl_test = deterministic('logl_test', Poisson(rate=rate).log_prob(test_y).sum() * factor_test_ll)
        if alternative_test_data is not None:
            alt_rate = jnp.exp(adjusted_onehotted_X_alt_test @ w)
            logl_alt_test = deterministic('logl_alt_test', Poisson(rate=alt_rate).log_prob(alt_test_y).sum() * factor_alt_test_ll)
            # gain = deterministic('logl_gain', logl_test - logl_alt_test)

    if use_mcmc:
        num_mc_samples = num_mc_samples * thinning_factor
        num_chains = 4
        mcmc = MCMC(NUTS(model), num_warmup=num_mc_samples//8, num_samples=num_mc_samples//num_chains, num_chains=num_chains, progress_bar=False, thinning=thinning_factor)
        mcmc.run(rng_key)
        samples = mcmc.get_samples(group_by_chain=True)
        # for var in ['logl_test', 'logl_alt_test', 'logl_gain']:
        for var in ['logl_test', 'logl_alt_test']:
            if var in samples.keys():
                rhat = split_gelman_rubin(samples[var])
                if rhat >= 1.01:
                    print(f"!!!! log-likelihood mcmc chains for {var} did not converge: Rhat = {rhat}")
        samples = mcmc.get_samples()
    else:
        samples = Predictive(model, num_samples=num_mc_samples, parallel=False)(rng_key)

    if alternative_test_data is not None:
        return samples['logl_test'], samples['logl_alt_test']#, samples['logl_gain']
    return samples['logl_test']

import os
import pickle

def load_params(prefix, center, args, **kwargs):
    """
    Loads parameters.
    """
    stored_model_name = filenamer(prefix, center, args, **kwargs)
    stored_model_path = f"{os.path.join(args.stored_model_dir, stored_model_name)}"

    with open(f"{stored_model_path}.p", "rb") as f:
        twinify_result = pickle.load(f)

    param_dict = twinify_result.model_params

    return param_dict
