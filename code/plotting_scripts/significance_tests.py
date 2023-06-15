import scipy.stats
import numpy as np
import pandas as pd

p_star_levels = np.array([.05, .01, .001])

def annotate_significance(p_vals, keys, labels = None):
    if labels is None:
        labels = keys
    new_labels = [str(label) + "*"*np.sum(p_vals[key] <= p_star_levels) for key, label in zip(keys, labels)]
    return new_labels


def compute_ranked_t_test_p(first_df: pd.DataFrame, second_df: pd.DataFrame, trim: float = 0., alternative: str = 'greater', log_threshold: int = -50):
    """ Computes p-value for combined lls differing from local-only lls using the ranked Welch t-test (for unequal variances).
    
    This makes no assumptions on normality of data.

    Arguments:
        first_df: Dataframe holding samples for a single center in column "lls" as the first sample group
        second_df: Dataframe holding samples for a single center in column "lls" as the second sample group
        trim: Amount of outliers to trim (at each end)
        alternative: Hypothesis to test. Defaults to combined being greater than local only.
        log_threshold: Equality threshold for ranking, i.e., x and y have equal rank if |x - y| <= 10**log_threshold
    """
    FIRST_KEY = "__first#!"
    SECOND_KEY = "__second#!"

    filtered_df = pd.concat((first_df.assign(source=FIRST_KEY), second_df.assign(source=SECOND_KEY)))

    ranks = scipy.stats.rankdata(np.round(filtered_df['lls'], -log_threshold))
    ranked_df = filtered_df

    # sorted_df = filtered_df.sort_values(by='lls')
    # ranks = np.cumsum(sorted_df['lls'].diff() > 10**(log_threshold))
    # ranked_df = sorted_df

    ranked_df['rank'] = ranks

    return scipy.stats.ttest_ind(
        ranked_df[ranked_df.source == FIRST_KEY]["rank"], ranked_df[ranked_df.source == SECOND_KEY]["rank"],
        equal_var=False, trim=trim, alternative=alternative
    ).pvalue
