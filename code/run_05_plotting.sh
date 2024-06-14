#!/bin/bash
# Script to create all figure plots.

source paths.sh
set -v

#Fig 1.:
python plotting_scripts/plot_ll_full_downstream.py \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
   --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
   --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
   --test_type avg_ll --max_center_size 0.1 --epsilon=1.0 --plot_without_barts

#Fig 2:
python plotting_scripts/plot_ll_over_number_of_centers.py all \
    --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
    --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
    --max_center_size 0.1 --test_type avg_ll --plot_variant min_and_max_box --plot_without_barts

#Fig 3:
python plotting_scripts/plot_ll_over_size_of_centers.py Newcastle \
    --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
    --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
    --test_type avg_ll --plot_variant single_box

#Fig 4:
python plotting_scripts/plot_ll_skewed_Newcastle_unadjusted.py \
    --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
    --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
    --test_type avg_ll

#Fig 5:
python plotting_scripts/plot_ll_over_size_of_subgroup.py \
    --test_data_path "${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv.B_SouthAsian" \
    --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/skewed/SouthAsian/testonly/reps100/" \
    --test_type avg_ll

#Fig 6:
python plotting_scripts/plot_ll_full_downstream_over_epsilons.py \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
   --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
   --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
   --test_type avg_ll --max_center_size 0.1 --epsilon 0.5 1.0 2.0 4.0 --plot_without_barts

#Fig S1:
python plotting_scripts/plot_ll_full_downstream.py \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
   --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
   --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
   --test_type avg_ll --max_center_size 0.1 --epsilon=2.0 --plot_without_barts

#Fig S2:
python plotting_scripts/plot_ll_over_number_of_centers.py all \
    --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
    --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
    --max_center_size 0.1 --test_type avg_ll --plot_variant min_and_max_box --plot_without_barts \
    --epsilon=2.0

#Fig S3:
python plotting_scripts/plot_ll_full_downstream.py \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
   --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
   --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
   --test_type avg_ll --max_center_size 0.1 --epsilon=1.0

#Fig S4:
python plotting_scripts/plot_ll_over_number_of_centers.py all \
    --lls_path "${UKB_BASE_FOLDER}/synthetic_data/lls_over_num_shared/reps100/" \
    --train_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080.csv" \
    --test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv" \
    --max_center_size 0.1 --test_type avg_ll --plot_variant all_separate_box
