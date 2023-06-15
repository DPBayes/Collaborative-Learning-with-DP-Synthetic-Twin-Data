#!/bin/sh

set -e

conda create -y --name collaborative-learning-with-syn-data python=3.8
source activate collaborative-learning-with-syn-data

pip install pandas==1.3.5 parsimonious==0.8.1 joblib==1.1.0 tqdm numpy==1.22.4 numpyro==0.8.0 scikit-learn==1.1.0 seaborn==0.12.2 scipy==1.7.3 matplotlib==3.4.3 statsmodels==0.12.2 jax[cpu]==0.2.22
pip install git+https://github.com/DPBayes/twinify.git@e326afec8bdef1a418d270984331c89878c4eaf1#egg=twinify
