import os

base_path = os.environ['UKB_BASE_FOLDER']
raw_data_path = os.path.join(base_path, 'original_data')
processed_data_path = os.path.join(base_path, 'processed_data')
synthetic_data_path = os.path.join(base_path, 'synthetic_data')
ukb_data_id = os.environ['UKB_DATA_ID']
ukb_project_id = os.environ['UKB_PROJECT_ID']

encodings = os.path.join(raw_data_path, 'encoding.ukb')
withdrawals = os.path.join(raw_data_path, 'latest_withdrawals.csv')
covid_data = os.path.join(raw_data_path, 'covid19_results.tsv')
base_data = os.path.join(raw_data_path, f'ukb{ukb_data_id}.csv')
base_data_ukb_format = os.path.join(raw_data_path, f'ukb{ukb_data_id}.enc_ukb')
column_infos = os.path.join(raw_data_path, 'columns.pickle')

model_one_full_span_data = os.path.join(processed_data_path, 'model_one_full_span_data.csv')
model_one_covid_tested_data = os.path.join(processed_data_path, 'model_one_covid_tested_data.csv')
