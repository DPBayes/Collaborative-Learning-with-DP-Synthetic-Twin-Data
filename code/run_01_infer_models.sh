#!/bin/bash
# Script for inferring generative models for all experiments on a system WITHOUT SLURM support.
# If SLURM is available, use slurm_01_infer_models.sh instead.


## Parameters to set manually
n_epochs=4000
clipping_threshold=2.0


source paths.sh

set -e # stop script if any command errors
source activate collaborative-learning-with-syn-data
export PYTHONUNBUFFERED=1

param_file="infer_models_params.txt"
num_jobs=$(wc -l < ${param_file})

output_dir="${UKB_BASE_FOLDER}/synthetic_data/models" # the directory to store the twinify output
mkdir -p $output_dir


############## Run twinify
numpyro_model_path="twinify_models/model1.py"

for (( n=1; n<=$num_jobs; n++ ))
do
    infer_seed=`sed -n "${n} p" ${param_file} | awk '{print $1}'` # Get seed
    eps=`sed -n "${n} p" ${param_file} | awk '{print $2}'` # Get epsilon
    max_center_size=`sed -n "${n} p" ${param_file} | awk '{print $3}'` # Get center size subsampling ratio

    # derived variables
    input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080$max_center_size_sfx.csv"

    max_center_size_arg="--max_center_size=$max_center_size"
    max_center_size_sfx="_max$max_center_size"
    if [ -z $max_center_size ]; then
        max_center_size_arg="";
        max_center_size_sfx="";
    fi

    echo "eps=${eps} infer_seed=${infer_seed} max_center_size=${max_center_size}"
    python experiment_scripts/infer_models.py $input_data_path $numpyro_model_path $output_dir --epsilon=$eps --seed=$infer_seed \
        --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold $max_center_size_arg
done
