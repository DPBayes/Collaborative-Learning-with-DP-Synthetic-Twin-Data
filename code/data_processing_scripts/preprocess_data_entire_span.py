""" reads UKB assessment base data CSV and covid test result CSV and produces the final preprocessed CSV used in experiments """

import numpy as np
import pandas as pd
from ukb_parsing import Columns
import datetime, sys

sys.path.append("../")
import paths

columns = Columns.load(paths.column_infos)

# # monkey-patch for pd.NA: Equality with anything should return False, not NA!
# type(pd.NA).__eq__ = lambda a, b: False # doesn't work, though....

date_columns = columns.get_all_of_type('DATE')
column_dtypes = columns.get_type_map({
    'DATE': str, 'FLOAT': 'float', 'INTEGER': 'Int32', 'STRING': str
})
del column_dtypes["eid"]
base_data_as_loaded = pd.read_csv(paths.base_data, sep=',', dtype=column_dtypes, parse_dates=date_columns, index_col=0)
covid_data_as_loaded = pd.read_csv(paths.covid_data, sep='\t', parse_dates=['specdate'], dtype={'result': bool, 'origin': bool}, index_col=0)
try:
    withdrawals = pd.read_csv(paths.withdrawals, header=None)
except FileNotFoundError:
    withdrawals = None

# extracting relevant information from covid data
covid_data = covid_data_as_loaded
covid_data = covid_data[['result', 'origin']]
covid_data.columns=['covid_test_result', 'covid_test_in_hospital']
covid_data = covid_data.assign(covid_test_positive_in_hospital = lambda x: (x['covid_test_result'] == 1) & (x['covid_test_in_hospital'] == 1))
covid_data= covid_data.groupby(['eid']).aggregate({
    'covid_test_result': 'max',
    'covid_test_positive_in_hospital': 'max'
})
assert (~covid_data['covid_test_result'] & covid_data['covid_test_positive_in_hospital']).sum() == 0

# remove participant that withdrew consent
base_data = base_data_as_loaded
if withdrawals is not None:
    base_data = base_data.drop(withdrawals.values.flatten(), axis='index')

# remove Stockport pilot participants
baseline_assessment_center_column = columns.get_column_idxs('UK Biobank assessment centre')[columns.BASELINE_ASSESSMENT][0]
pilot_assessment_center = columns.get_column_encoding(baseline_assessment_center_column).encode('Stockport (pilot)')
non_pilot_indices = (base_data[baseline_assessment_center_column] != pilot_assessment_center) | base_data[baseline_assessment_center_column].isna()
base_data = base_data[non_pilot_indices]

# remove participants that died before start of study
date_of_death_column = columns.get_column_idxs('Date of death')[0][0]
not_died_before_start = (base_data[date_of_death_column] > pd.to_datetime("2018-01-31")) | base_data[date_of_death_column].isna()
base_data = base_data[not_died_before_start]


##### convert raw UKB values into categories used by study ######
processed_data = pd.DataFrame()

def column_id(title):
    return columns.get_column_idxs(title)[0][0]

def map_column_values(dest_df, source_df, column_title, new_title, mapping, decode_passthrough=False):
    """ Map encoded values in a single column according to a dictionary mapping (specified using decoded values). """
    cid = column_id(column_title)
    encoding = columns.get_column_encoding(cid)
    return dest_df.assign(
        **{new_title: source_df[cid].map(
            lambda x: encoding.decode(x, transcode_map=mapping, passthrough=decode_passthrough)
        ).astype('category')}
    )

# map qualifications as in paper
quali_cid = column_id('Qualifications')
quali_encoding = columns.get_column_encoding(quali_cid)
qualifications_map = {
    'College or University degree': 'College or University degree',
    'A levels/AS levels or equivalent': 'A levels/AS levels or equivalent',
    'O levels/GCSEs or equivalent': 'O levels/GCSEs/CSEs',
    'CSEs or equivalent': 'O levels/GCSEs/CSEs',
    'NVQ or HND or HNC or equivalent': 'Others',
    'Other professional qualifications eg: nursing, teaching': 'Others',
    'None of the above': 'None of the above',
    'Prefer not to answer': pd.NA, #'Prefer not to answer'
    pd.NA: pd.NA
}
processed_data = processed_data.assign(
    education = base_data[columns.get_column_idxs('Qualifications')[0]].min(1).astype('Int64').map(
        lambda x: quali_encoding.decode(x, transcode_map=qualifications_map)
    ).astype("category")
)

# map ethnicity as in paper
ethnicity_map = {
    'British': 'White British',
    'Irish': 'White Irish',
    'Any other white background': 'White Other',
    'White': 'White Other',

    'Mixed': 'Mixed',
    'White and Black Caribbean': 'Mixed',
    'White and Black African': 'Mixed',
    'White and Asian': 'Mixed',
    'Any other mixed background': 'Mixed',

    'Indian': 'South Asian',
    'Pakistani': 'South Asian',
    'Bangladeshi': 'South Asian', # 'Other South Asian'
    # note(lukas): the following could potentially also be 'Other', but the 'Other South Asian' group of paper is larger than only 'Bangladeshi';
    # seems like anything Asian other than Chinese is considered South Asian by the authors
    'Any other Asian background': 'South Asian',
    'Asian or Asian British': 'South Asian', #?

    'Black or Black British': 'Black',
    'Caribbean': 'Black',
    'African': 'Black',
    'Any other Black background': 'Black',

    'Chinese': 'Chinese',
    'Other ethnic group': 'Other',

    'Do not know': pd.NA,
    'Prefer not to answer': pd.NA,
    pd.NA: pd.NA
}
processed_data = map_column_values(processed_data, base_data, 'Ethnic background', 'ethnicity', ethnicity_map)

# map country of birth
cob_map = {
    'England': 'UK and Ireland',
    'Wales': 'UK and Ireland',
    'Scotland': 'UK and Ireland',
    'Northern Ireland': 'UK and Ireland',
    'Republic of Ireland': 'UK and Ireland',
    'Elsewhere': 'Elsewhere',
    'Do not know': pd.NA,
    'Prefer not to answer': pd.NA,
    pd.NA: pd.NA
}
processed_data = map_column_values(processed_data, base_data, 'Country of birth (UK/elsewhere)', 'country_of_birth', cob_map)

# map longstanding illness
processed_data = processed_data.assign(
    longstanding_illness = base_data[column_id('Long-standing illness, disability or infirmity')].map(
        lambda x: x if x is not pd.NA and x >= 0 else pd.NA
    )
)

# map number of chronic health conditions illnesses
# todo(lumip): this will need some more work...

# map overall health status
processed_data = processed_data.assign(
    overall_health = base_data[column_id('Overall health rating')].map(
        lambda x: x if x is not pd.NA and x >= 0 else pd.NA
    )
)

# map smoking habits
processed_data = processed_data.assign(
    smoking_status = base_data[column_id('Smoking status')].map(
        lambda x: x if x is not pd.NA and x >= 0 else pd.NA
    )
)

# map BMI to categories
processed_data = processed_data.assign(
    bmi = pd.cut(
        base_data[column_id('Body mass index (BMI)')],
        [0, 18.5, 25, 30, 1000], right=False, include_lowest=True,
        labels=['underweight', 'normal weight', 'overweight', 'obese']
    )
)

# map alcohol consumption
alc_freq_cid = column_id('Alcohol intake frequency.')
alc_form_cid = column_id('Former alcohol drinker')
alc_freq_encoding = columns.get_column_encoding(alc_freq_cid)
alc_form_encoding = columns.get_column_encoding(alc_form_cid)
def map_drinking_habit(x):
    if x[alc_freq_cid] is pd.NA or x[alc_freq_cid] < 0: return pd.NA
    if x[alc_freq_cid] == alc_freq_encoding.encode('Never'):
        if x[alc_form_cid] is not pd.NA and x[alc_form_cid] == alc_form_encoding.encode('Yes'): return 'Never (former drinker)'
        else: return 'Never'
    else:
        return alc_freq_encoding.decode(x[alc_freq_cid])

processed_data = processed_data.assign(
    alcohol_consumption = base_data.apply(map_drinking_habit, axis='columns').astype('category')
)

# map employment status
employment_map = {
    'In paid employment or self-employed': 'In paid employment or self-employed',
    'Retired': 'Retired',
    'Looking after home and/or family': 'Looking after home and/or family',
    'Unable to work because of sickness or disability': 'Unable to work because of sickness or disability',
    'Unemployed': 'Unemployed',
    'Doing unpaid or voluntary work': 'Other',
    'Full or part-time student': 'Other',
    'None of the above': 'Other',
    'Prefer not to answer': pd.NA,
    pd.NA: pd.NA
}
processed_data = map_column_values(processed_data, base_data, 'Current employment status', 'employment_status', employment_map)

# map manual labour
manual_labour_map = {
    'Never/rarely': 'Non-manual',
    'Sometimes': 'Non-manual',
    'Usually': 'Manual',
    'Always': 'Manual',
    'Do not know': pd.NA,
    'Prefer not to answer': pd.NA,
    pd.NA: pd.NA
}
processed_data = map_column_values(processed_data, base_data, 'Job involves heavy manual or physical work', 'manual_occupation', manual_labour_map)
# todo(lumip): how to deal with "Not in employment here"? currently gives NAs, which will bite us when we do dropna later on

# identify healthcare workers...
# soc2000 codes: https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/standardoccupationalclassificationsoc/socarchive/soc2000tcm771813161.pdf
def filter_healthcare_worker(soc):
    if soc is pd.NA: return pd.NA
    return (soc >= 2200 and soc < 2300) or (soc >= 3200 and soc < 3300) or \
        (soc >= 1180 and soc < 1190) or (soc >= 6110 and soc < 6120) or \
        (soc == 9221) or (soc == 4211)
processed_data = processed_data.assign(
    is_healthcare_worker=base_data[column_id('Job code at visit')].map(filter_healthcare_worker)
)

# map housing tenure
housing_map = {
    'Own outright (by you or someone in your household)': 'Own',
    'Own with a mortgage': 'Own',
    'Rent - from local authority, local council, housing association': 'Rent/others',
    'Rent - from private landlord or letting agency': 'Rent/others',
    'Pay part rent and part mortgage (shared ownership)': 'Rent/others',
    'Live in accommodation rent free': 'Rent/others',
    'None of the above': 'Rent/others',
    'Prefer not to answer': pd.NA,
    pd.NA: pd.NA,
    -1: pd.NA # for some reason, there are two rows that have a value of -1 (usually, 'Do not know'), which the encoding says does not exist...
}
processed_data = map_column_values(processed_data, base_data, 'Own or rent accommodation lived in', 'housing_tenure', housing_map, decode_passthrough=True)

# map urban/rural
urban_map = {
    'England/Wales - Urban - sparse': 'Urban',
    'England/Wales - Urban - less sparse': 'Urban',
    'Scotland - Other Urban Area': 'Urban',
    'Scotland - Large Urban Area': 'Urban',

    'Scotland - Very Remote Rural': 'Rural',
    'Scotland - Remote Rural': 'Rural',
    'Scotland - Accessible Rural': 'Rural',
    'England/Wales - Village - less sparse': 'Rural',
    'England/Wales - Hamlet and Isolated dwelling - sparse': 'Rural',
    'England/Wales - Village - sparse': 'Rural',
    'England/Wales - Hamlet and Isolated Dwelling - less sparse': 'Rural',
    'Scotland - Very Remote Small Town': 'Rural',
    'England/Wales - Town and Fringe - sparse': 'Rural',
    'England/Wales - Town and Fringe - less sparse': 'Rural',
    'Scotland - Remote Small Town': 'Rural',
    'Scotland - Accessible Small Town': 'Rural',

    'Postcode not linkable': pd.NA,
    pd.NA: pd.NA
}
processed_data = map_column_values(processed_data, base_data, 'Home area population density - urban or rural', 'urban/rural', urban_map)

# map number of people in house
def map_number_in_household(x):
    if x is pd.NA or x < 0: return pd.NA
    return x if x < 4 else 4

number_in_household_cid = column_id('Number in household')
processed_data = processed_data.assign(
    number_in_household=base_data[number_in_household_cid].map(
        map_number_in_household
    )
)

# make age groups
age_cid = column_id('Age at recruitment')
processed_data = processed_data.assign(
    age_group = pd.cut(base_data[age_cid], [40, 45, 50, 55, 60, 65, 70, 200], right=False)
)


# make deprivation quartiles
deprivation_cid = column_id('Townsend deprivation index at recruitment')
processed_data = processed_data.assign(
    deprivation = pd.qcut(base_data[deprivation_cid], 4, labels=['1st', '2nd', '3rd', '4th']),
)

# get assessment center
ac_encoding = columns.get_column_encoding(baseline_assessment_center_column)
processed_data = processed_data.assign(
    assessment_center = base_data[baseline_assessment_center_column].map(
        lambda x: ac_encoding.decode(x)
    )
)

# get sex
sex_cid = column_id('Sex')
sex_encoding = columns.get_column_encoding(sex_cid)
processed_data = processed_data.assign(
    sex = base_data[sex_cid].map(
        lambda x: sex_encoding.decode(x)
    )
)

# drop incomplete data items
processed_data_with_missing = processed_data
processed_data = processed_data_with_missing.dropna(
    subset=set(processed_data.columns) - {'manual_occupation', 'is_healthcare_worker'}
)
#todo(lumip): drop those that are employed according to employment_status but have is_healthcare_worker is NA

# merge with covid19 data
processed_data = processed_data.join(covid_data, how='left')
processed_data = processed_data.assign(
    covid_test_present = ~processed_data['covid_test_result'].isna()
)

# extract model one features
all_model_columns = ['ethnicity', 'deprivation', 'education', 'covid_test_present', 'covid_test_result', 'covid_test_positive_in_hospital']
model_one_columns = ['age_group', 'sex', 'assessment_center'] + all_model_columns
model_one_data = processed_data[model_one_columns]

#model_one_data.to_csv(paths.model_one_full_span_data)

# pick only the covid tested
model_one_tested_data = model_one_data[~(model_one_data.covid_test_result).isna()]

model_one_tested_data.to_csv(paths.model_one_covid_tested_data)
