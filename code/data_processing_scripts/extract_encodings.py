""" extracts encodings from UKB metadata / encoding files into a python readable format"""

import ukb_parsing, sys

sys.path.append("../")
import paths

print("reading ukb encodings file (may take a while)")
with open(paths.encodings, "r") as f:
    encodings = ukb_parsing.parse_ukb_encoding_file(f)

print("reading encoding ids from data header")
with open(paths.base_data_ukb_format, "r") as f:
    columns = ukb_parsing.Columns.from_data(f, encodings)

columns.save(paths.column_infos)
