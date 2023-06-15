#!/bin/sh
# This script bundles operations to decrypt, convert and filter UKB data to
# create all data files in `original_data` and `processed_data` that are required for running the experiments.

set -e

PASSWD=$1
RAW_DATA_FOLDER="${UKB_BASE_FOLDER}/original_data"
PROCESSED_DATA_FOLDER="${UKB_BASE_FOLDER}/processed_data"

FILENAME="${RAW_DATA_FOLDER}/ubk${UKB_DATA_ID}"

# decrypt and convert native UKB data to csv
ukbunpack "${FILENAME}.enc" ${PASSWD}
ukbconv "${FILENAME}.enc_ukb" csv
ukbconv "${FILENAME}.enc_ukb" docs # creates docs for manually interpreting the data

# create python interpretable map of UKB encoding
python data_processing_scripts/extract_encodings.py

# map UKB encodings and split train test data
python data_processing_scripts/preprocess_data_entire_span.py

TEST_SPLIT_PCT=20
python data_processing_scripts/split_train_test.py --test-data-pct=$TEST_SPLIT_PCT

# limit center sizes by subsampling (Fig. 1-3)
TEST_SPLIT_SFX="_test020"
TRAIN_SPLIT_SFX="_train080"
TRAIN_DATA_PATH="${PROCESSED_DATA_FOLDER}/model_one_covid_tested_data${TRAIN_SPLIT_SFX}.csv"
TEST_DATA_PATH="${PROCESSED_DATA_FOLDER}/model_one_covid_tested_data${TEST_SPLIT_SFX}.csv"
python data_processing_scripts/subsample_center_data.py ${TRAIN_DATA_PATH} "${PROCESSED_DATA_FOLDER}/model_one_covid_tested_data${TRAIN_SPLIT_SFX}_max0.1.csv" 0.1
python data_processing_scripts/subsample_center_data.py ${TRAIN_DATA_PATH} "${PROCESSED_DATA_FOLDER}/model_one_covid_tested_data${TRAIN_SPLIT_SFX}_max0.2.csv" 0.2
python data_processing_scripts/subsample_center_data.py ${TRAIN_DATA_PATH} "${PROCESSED_DATA_FOLDER}/model_one_covid_tested_data${TRAIN_SPLIT_SFX}_max0.5.csv" 0.5

# create skewed artificial center and corresponding test set (Fig 5)
python data_processing_scripts/create_uniform_centers.py ${TEST_DATA_PATH} ${TEST_DATA_PATH} 2 --separate_outputs # split test data into two
FIRST_HALF_TEST_DATA="${TEST_DATA_PATH}.A"
SECOND_HALF_TEST_DATA="${TEST_DATA_PATH}.B"
python data_processing_scripts/create_skewed_centers.py ${FIRST_HALF_TEST_DATA} "${PROCESSED_DATA_FOLDER}/model_one_covid_tested_data${TEST_SPLIT_SFX}_splitA_skewedSouthAsian.csv" --feature "ethnicity" --category "South Asian" # create skewed center data
python data_processing_scripts/create_feature_category_test_set.py ${SECOND_HALF_TEST_DATA} "${SECOND_HALF_TEST_DATA}_SouthAsian"  --feature "ethnicity" --category "South Asian" # filter test set data to relevant category
