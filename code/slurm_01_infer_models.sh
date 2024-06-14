#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=8G

# Script for inferring generative models for all experiments on a system WITH SLURM support.
# If SLURM is NOT available, use slurm_01_infer_models.sh instead.


## Parameters to set manually
n_epochs=4000
clipping_threshold=2.0


module load anaconda
source paths.sh

set -e # stop script if any command errors
source activate collaborative-learning-with-syn-data
echo $(which pip)
echo $(which python)
export PYTHONUNBUFFERED=1

n=$SLURM_ARRAY_TASK_ID
param_file="infer_models_params.txt"
infer_seed=`sed -n "${n} p" ${param_file} | awk '{print $1}'` # Get seed
eps=`sed -n "${n} p" ${param_file} | awk '{print $2}'` # Get epsilon
max_center_size=`sed -n "${n} p" ${param_file} | awk '{print $3}'` # Get center size subsampling ratio

output_dir="${UKB_BASE_FOLDER}/synthetic_data/models" # the directory to store the twinify output
mkdir -p $output_dir

# derived variables
max_center_size_arg="--max_center_size=$max_center_size"
max_center_size_sfx="_max$max_center_size"
if [ -z $max_center_size ]; then
    max_center_size_arg="";
    max_center_size_sfx="";
fi

############## Run twinify
numpyro_model_path="twinify_models/model1.py"

input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data_train080$max_center_size_sfx.csv"
srun python experiment_scripts/infer_models.py $input_data_path $numpyro_model_path $output_dir --epsilon=$eps --seed=$infer_seed \
    --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold $max_center_size_arg
