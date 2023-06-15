#!/bin/bash
#SBATCH -c 10
#SBATCH --time=1-12:00:00
#SBATCH --mem=70G

# Script for running the analysis task combining local data and synthetic data for all experiments on a system WITH SLURM support.
# If SLURM is NOT available, use run_03_lls_over_num_shared.sh instead.
# Leave unchanged to produce results for Fig 1-4. Set use_skewed_local=1 and test_skew_only=1 below to produce results for Fig 5.


## Parameters to set manually
n_epochs=4000
clipping_threshold=2.0
num_reps="100"
use_skewed_local=0 # use skewed center data as the local center; set to 1 for Fig 5, 0 for Fig 1-4
skew_category="SouthAsian" # ignored if $use_skewed_local is 0
test_skew_only=1 # use the test set reduced to the category skewed - ignored if $use_skewed_local is 0; set to 1 for Fig 5, 0 for Fig 1-4


module load anaconda
source paths.sh

set -e # stop script if any command errors
source activate collaborative-learning-with-syn-data
export PYTHONUNBUFFERED=1

test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv"
test_output_sfx=""

input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080$max_center_size_sfx.csv"

params_file="lls_run_params.txt"
if [ "$use_skewed_local" -eq "1" ]; then
    input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020_splitA_skewed${skew_category}.csv"
    test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020_split.csv.B"
    test_output_sfx="/skewed/${skew_category}/testall"
    params_file="lls_skewed_run_params.txt"

    if [ "$test_skew_only" -eq "1" ]; then
        test_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_test020.csv.B_${skew_category}"
        test_output_sfx="/skewed/${skew_category}/testonly"
    fi
fi

n=$SLURM_ARRAY_TASK_ID
center=`sed -n "${n} p" ${params_file} | awk '{print $1}'`
eps=`sed -n "${n} p" ${params_file} | awk '{print $2}'` # Get epsilon
max_center_size=`sed -n "${n} p" ${params_file} | awk '{print $3}'` # Get center size subsampling ratio

max_center_size_arg="--max_center_size=$max_center_size"
max_center_size_sfx="_max$max_center_size"
if [ -z $max_center_size ]; then
    max_center_size_arg="";
    max_center_size_sfx="";
fi

path_base="${UKB_BASE_FOLDER}/synthetic_data"
syn_data_dir="${path_base}/twin_data/"

output_dir="${path_base}/lls_over_num_shared${test_output_sfx}/reps${num_reps}/"
mkdir -p $output_dir

srun python experiment_scripts/compute_ll_over_number_of_centers.py $center --train_data_path $input_data_path --test_data_path $test_data_path \
    --syn_data_path $syn_data_dir --output_path $output_dir --epsilon=$eps --k=16 --num_epochs=$n_epochs \
    --clipping_threshold=$clipping_threshold $max_center_size_arg --num_repetitions=$num_reps --num_processes=10 --num_mc_samples=100
