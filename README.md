# Collaborative Learning from Distributed Data with Differentially Private Synthetic Twin Data

## Python Environment

Run the file `setup_python_environment.sh` to create and set up an anaconda/miniconda environment named `collaborative-learning-with-syn-data`.

## Code Layout

The code is organized as follows:

```
code/
    - data_processing_scripts/  # contains scripts to convert, filter and preprocess data obtained from the UK Biobank
    - experiment_scripts/       # contains scripts to produce results of the paper experiments
    - plotting_scripts/         # contains scripts to create the plots shown in the paper's figures
```

`code/` also contains the following driver scripts:

- `paths_template.sh`: A template file to set a number of environmental variables. See the next section in this file for further information.
- `create_param_files.py`: Run to create list of parameters required by the following scripts.
- `run_01_infer_models.sh`: Run to infer parameters of the generative models for all paper experiments.
- `run_02_generative_twin_data.sh`: Run to generative synthetic twin data from the generative models for all paper experiments.
- `run_03_lls_over_num_shared.sh`: Run to perform the analysis task on combined local and synthetic data for all paper experiments.
- `run_04_plotting.sh`: Run to create the plots in the paper.

If you have access to a system equipped with the SLURM workload manager, use the files prefixed `slurm_` instead of `run_`.

Note: `run_03_lls_over_num_shared.sh` (or its corresponding slurm file) must be edited manually after being run once to create the results
for Figure 5 (see instructions in the file).

## Data Layout

All scripts require the following environmental variables to be set:

- `UKB_BASE_FOLDER`: Path to the directory that contains all data
- `UKB_DATA_ID`: Numerical data identifier designated by UKB
- `UKB_PROJECT_ID`: Numerical project identifier designated by UKB

Please set the corresponding values in `code/paths_template.sh` and rename or copy the file to `code/paths.sh`.

All scripts make the following assumptions about the layout of the data directory

```
$UKB_BASE_FOLDER/
    - original_data/  # contains files directly obtained from UK Biobank and files derived using UKB conversion programs
    - processed_data/ # contains filtered and preprocessed UK Biobank data, prepared for use in the experiments
    - synthetic_data/ # contains learned generative models and synthetic twin data
```

`$UKB_BASE_FOLDER/original_data/` contains the following files, where XYZ is a placeholder for your UKB designated data identifier.

- `ukbXYZ.enc`: Encrypted raw UKB main data set.
    How to get: Apply to UKB with data fields listed in .... .
- `ukbXYZ.enc_ukb`: Decrypted raw UKB data in UKB's native format.
    How to get: `ukbunpack ukbXYZ.enc <key>` , where <key> is provided by UKB via e-mail.
- `ukbXYZ.csv`: Raw UKB data in csv format.
    How to get: `ukbconv ukbXYZ.enc_ukb csv`
- `encoding.ukb`: UKB encoding specification file.
    How to get: Download from UKB: `wget -nd  biobank.ctsu.ox.ac.uk/crystal/util/encoding.ukb`
- `columns.pickle`: Mapping of encoded values for columns in the data in our own Python format.
    How to get: Run `code/data_processing_scripts/extract_encodings.py`
- `wXYZ_<YYYYMMDD>.csv`: List of participants that have withdrawn after obtaining original data set at given date.
    How to get: Regularly provided by UKB via e-mail.
- `latest_withdrawals.csv`: Last complete list of all withdrawn participants.
    How to get: Symlink to the latest wXYZ_*.csv file.
- `covid19_results.tsv`: Covid19 test data downloaded from UKB data portal.
    How to get: Apply for access to UKB and download using data portal (cf. Sec. 5 in Data Access Guide: https://biobank.ctsu.ox.ac.uk/~bbdatan/Accessing_UKB_data_v2.3.pdf )

`$UKB_BASE_FOLDER/processed_data/` contains the following files:

- `model_one_full_span_data.csv`: UK Biobank data reduced to relevant fields and converted to interpretable fields, for all individuals.
    How to get: Run `code/data_processing_scripts/preprocess_data_entire_span.py`
- `model_one_covid_tested_data.csv`: Same as above, but only individuals for which at least one SARS-CoV-19 test result was present.
    How to get: Run `code/data_processing_scripts/preprocess_data_entire_span.py`
- `model_one_covid_tested_data_[train080|test020].csv`: The previous, split into train and test sets.
    How to get: Run `code/data_processing_scripts/split_train_test.py`
- `model_one_covid_tested_data_train080_maxRR.csv`: The train data split subsampled with a given ratio.
    How to get: Run `code/data_processing_scripts/subsample_center_data.py $UKB_BASE_FOLDER/processed_data/model_one_covid_tested_data_train080.csv $UKB_BASE_FOLDER/processed_data/ <subsample ratio, e.g., 0.2>`

## Data Setup

To populate the directories with the files initially required, follow the steps below:

1. Download `ukbXYZ.enc` and `covid19_results.tsv` following the instructinos provided by UKB.
2. Download `encoding.ukb` and the UKB file format conversion programs from https://biobank.ndph.ox.ac.uk/ukb/download.cgi (cf. https://biobank.ctsu.ox.ac.uk/crystal/exinfo.cgi?src=accessing_data_guide)
3. If you were provided with a withdrawal file, create the `latest_withdrawals.csv` symbolic link (or copy and rename the original file).
4. Ensure the environment variables listed in the beginning of this file are set and the UKB conversion programs are in PATH.
5. Run `code/run_00_process_data.sh <key>` , where <key> is provided by UKB via e-mail.
    - This script bundles operations to decrypt, convert and filter UKB data to create all data files in `original_data` and `processed_data` that are required for running the experiments.