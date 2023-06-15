#!/bin/bash
# Script to generate synthetic twin data for all experiments on a system WITHOUT SLURM support.
# If SLURM is available, use slurm_02_generate_twin_data.sh instead.


## Parameters to set manually
n_epochs=4000
clipping_threshold=2.0
num_synthetic_data_sets=100


source paths.sh

set -e # stop script if any command errors
source activate collaborative-learning-with-syn-data
export PYTHONUNBUFFERED=1

param_file="infer_models_params.txt"
num_jobs=$(wc -l < ${param_file})

# derived variables
base_path="${UKB_BASE_FOLDER}/synthetic_data"
twinify_result_dir="${base_path}/models" # the directory in which inferred model params are stored
twin_data_dir="${base_path}/twin_data/"  # the directory in which generated data will be stored
mkdir -p $twin_data_dir

############## Run twinify
numpyro_model_path="twinify_models/model1.py"

for (( n=1; n<=$num_jobs; n++ ))
do
    infer_seed=`sed -n "${n} p" ${param_file} | awk '{print $1}'` # Get seed
    eps=`sed -n "${n} p" ${param_file} | awk '{print $2}'` # Get epsilon
    max_center_size=`sed -n "${n} p" ${param_file} | awk '{print $3}'` # Get center size subsampling ratio

    # derived variables
    input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080${max_center_size_sfx}.csv"

    max_center_size_arg="--max_center_size=$max_center_size"
    max_center_size_sfx="_max$max_center_size"
    if [ -z $max_center_size ]; then
        max_center_size_arg="";
        max_center_size_sfx="";
    fi

    echo "eps=${eps} infer_seed=${infer_seed} max_center_size=${max_center_size}"
    python experiment_scripts/generate_twin_data.py $input_data_path $numpyro_model_path $twinify_result_dir --output_dir=$twin_data_dir \
        --epsilon=$eps --seed=$infer_seed --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold \
        --num_synthetic_data_sets=$num_synthetic_data_sets $max_center_size_arg --batch_generate
done
