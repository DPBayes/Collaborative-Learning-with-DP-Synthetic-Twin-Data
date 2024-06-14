#!/bin/sh

set -e

conda create -y --name collaborative-learning-with-syn-data python=3.8
source activate collaborative-learning-with-syn-data

pip install pandas==1.3.5 parsimonious==0.8.1 joblib==1.1.0 tqdm numpy==1.22.4 numpyro==0.11.0 scikit-learn==1.1.0 seaborn==0.12.2 scipy==1.7.3 matplotlib==3.4.3 statsmodels==0.12.2 jax[cpu]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install git+https://github.com/DPBayes/twinify.git@e326afec8bdef1a418d270984331c89878c4eaf1#egg=twinify
pip install --no-deps git+https://github.com/DPBayes/d3p.git@8c978db5be32b8361bd89ee43003d99ad864719b#egg=d3p