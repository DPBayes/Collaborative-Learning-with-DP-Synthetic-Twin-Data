""" utility to parse UKB encodings """

import parsimonious
import pickle
from functools import reduce
import numpy as np
import datetime
import pandas as pd

__all__ = ['parse_ukb_data_header', 'parse_ukb_encoding_file', 'Encoding', 'Columns']

ukb_base_grammar = r"""
    attribute_field = int_field / datetime_field / string_field
    int_field = "&i#" identifier "=" int
    int = ~"[+-]?[0-9]+"
    identifier = ~"[A-Z _]+"i
    string_field = "&s#" identifier "=" string
    string = ~"[^&\]]*"
    datetime_field = "&t#" identifier "=" datetime

    year = ~"[0-9]{4}"
    month = ~"[01]?[0-9]"
    day = ~"[0-3]?[0-9]"
    date = year "-" month "-" day
    hour = ~"[0-2]?[0-9]"
    minutesecond = ~"[0-5][0-9]"
    time = hour ":" minutesecond ":" minutesecond
    datetime = date ("T" time)?
    """

test_example = "[1&R#e1=[9&i#encoding_id=1&t#created=2020-10-2&t#updated=2020-10-2T20:01:58&s#name=Flag+indicating+Yes/True/Presence&i#coded_as=11&i#structure=1&i#generality=0&i#nmembers=2&R#c0=[5&i#ci=1&i#pi=0&s#va=1&i#se=1&s#mn=Yes]&R#c1=[5&i#ci=1&i#pi=0&s#va=1&i#se=1&s#mn=No]]]"
ukb_encoding_grammar = parsimonious.Grammar(
    r"""
    root = "[" int encoding_entries "]"
    encoding_entries = encoding_entry+
    encoding_entry = "&R#" encoding_id "=" encoding
    encoding_id = numeric_encoding_id / identifier
    numeric_encoding_id = "e" int
    encoding = "[" encoding_header encoding_values "]"
    encoding_header = int attribute_field+
    encoding_values = encoding_value_entry*
    encoding_value_entry = "&R#c" int "=" "[" encoding_value_attributes "]"
    encoding_value_attributes = int attribute_field+
    """ + ukb_base_grammar
)

ukb_data_header_grammar = parsimonious.Grammar(
    r"""
    root = "TREX[" int attribute_fields columns "]"
    attribute_fields = attribute_field*
    columns = column_info*

    column_info = "&R#f" int "=" "[" int column_attributes "]"
    column_attributes = attribute_field*
    """ + ukb_base_grammar
)


class Encoding:

    def __init__(self, attributes, codes):
        self.attributes = attributes
        self.codes = codes

        self.decode_map = {
            coding['va']: coding['mn'] for coding in self.codes
        }
        self.encode_map = {
            v: k for k, v in self.decode_map.items()
        }

    def __str__(self):
        return f"Encoding({self.id}, {self.url})"

    def __repr__(self):
        return str(self)

    def decode(self, code, transcode_map=None, passthrough=False):
        if code is pd.NA:
            return code
        try:
            v = self.decode_map[str(code)]
            if transcode_map:
                return transcode_map[v]
            else:
                return v
        except KeyError as e:
            if passthrough:
                return code
            raise e

    def encode(self, value, passthrough=False, dtype=int):
        if value is pd.NA:
            return pd.NA

        retval = value
        try:
            retval = self.encode_map[value]
        except KeyError as e:
            if not passthrough:
                raise e

        try:
            return dtype(retval)
        except:
            return retval


    @property
    def id(self):
        return self.attributes['encoding_id']

    @property
    def url(self):
        return f"https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id={self.id}"

    @staticmethod
    def Unknown(encoding_id):
        return Encoding({'encoding_id': encoding_id}, [])

class UKBFormatParserBase(parsimonious.NodeVisitor):

    def __init__(self, insert_spaces=True):
        self._insert_spaces = insert_spaces

    def generic_visit(self, node, visited_children):
        return visited_children or node.text

    def visit_numeric_encoding_id(self, node, visited_children):
        # numeric_encoding_id = "e" int
        return visited_children[1]

    def visit_int(self, node, visited_children):
        return int(node.text)

    def visit_identifier(self, node, visited_children):
        return node.text

    def visit_string(self, node, visited_children):
        return node.text.replace('+', ' ') if self._insert_spaces else node.text

    def visit_datetime(self, node, visited_children):
        return node.text

    def visit_attribute_field(self, node, visited_children):
        visited_children = visited_children[0]
        return (visited_children[1], visited_children[3])

class UKBDataHeaderParser(UKBFormatParserBase):
    """ Parser for header of UKB data format.

    Can be used to obtain readable title and encoding used by a column (as well as some other meta-information).
    """

    def __init__(self, insert_spaces=True, column_process_fn=None):
        """
        Args:
            insert_spaces: UKB format occasionally uses '+' instead of spaces. True to replace these, False to leave unchanged.
            column_process_fn: Optional post-processing function acceptions column_id (int), column_info (dict)
                and returns a processed dictionary. Can be used to filter out clutter.
        """
        super().__init__(insert_spaces)
        self._column_process_fn = column_process_fn

    def visit_column_info(self, node, visited_children):
        """ column_info = "&R#f" int "=" "[" int column_attributes "]" """
        column_id = visited_children[1]
        assert(isinstance(column_id, int))
        num_attributes = visited_children[4]
        assert(isinstance(num_attributes, int))
        attributes = visited_children[5]
        if num_attributes != len(attributes):
            raise Exception("Format error: Number of fields in a column descriptor does not match advertised count.")

        column_info = {k: v for k, v in attributes}
        if self._column_process_fn:
            column_info = self._column_process_fn(column_id, dict(**column_info))
        return column_id, column_info

    def visit_columns(self, node, visited_children):
        """ columns = column_info* """
        return visited_children

    def visit_root(self, node, visited_children):
        return visited_children[3]

    def visit_attribute_field(self, node, visited_children):
        k, v = super().visit_attribute_field(node, visited_children)
        if k == 'title':
            try:
                v = bytes.fromhex(v).decode('utf8')
            except: pass
        return k, v

class UKBEncodingParser(UKBFormatParserBase):
    """ Parser for UKB encoding.ukb file.
    """

    def visit_root(self, node, visited_children):
        num_encodings = visited_children[1]
        encodings = visited_children[2]
        assert(isinstance(num_encodings, int))
        if num_encodings != len(encodings):
            raise Exception("Format error: Number of encodings does not match advertised count.")
        return encodings

    def visit_encoding_entry(self, node, visited_children):
        encoding_id = visited_children[1]
        encoding = visited_children[3]
        assert(isinstance(encoding, Encoding))
        return encoding_id[0], encoding

    def visit_encoding_value_attributes(self, node, visited_children):
        num_attributes = visited_children[0]
        attributes = visited_children[1]
        assert(isinstance(num_attributes, int))
        if len(attributes) != num_attributes:
            raise Exception("Format error: Number of fields in a encoding value does not match advertised count.")

        attribute_dict = {
            key: val for key, val in attributes
        }
        return attribute_dict

    def visit_encoding_value_entry(self, node, visited_children):
        # encoding_value_entry = "&R#c" int "=" "[" encoding_value_attributes "]"
        entry_id = visited_children[1]
        attributes = visited_children[4]
        assert(isinstance(entry_id, int))
        assert(isinstance(attributes, dict))
        return entry_id, attributes

    def visit_encoding_values(self, node, visited_children):
        # encoding_values = encoding_value_entry*
        return [attributes for _, attributes in visited_children]

    def visit_encoding(self, node, visited_children):
        # encoding = "[" encoding_header encoding_values "]"
        attributes = visited_children[1]
        values = visited_children[2]
        assert(isinstance(attributes, dict))
        assert(isinstance(values, list))
        return Encoding(attributes, values)

    def visit_encoding_header(self, node, visited_children):
        num_attributes = visited_children[0]
        attributes = visited_children[1]
        assert(isinstance(num_attributes, int))
        attribute_dict = {
            key: val for key, val in attributes
        }
        return attribute_dict

def parse_ukb_data_header(f, insert_spaces=True, column_process_fn=None):
    """ Parses header of UKB data format.

    Can be used to obtain readable title and encoding used by a column (as well as some other meta-information).

    Args:
        insert_spaces: UKB format occasionally uses '+' instead of spaces. True to replace these, False to leave unchanged.
        column_process_fn: Optional post-processing function acceptions column_id (int), column_info (dict)
            and returns a processed dictionary. Can be used to filter out clutter.
    Returns:
        list of tuples (column_id, column_info)
    """
    f.seek(0)
    header_line = f.readline().strip()
    parse_tree = ukb_data_header_grammar.parse(header_line)

    return UKBDataHeaderParser(insert_spaces, column_process_fn).visit(parse_tree)



def parse_ukb_encoding_file(f, insert_spaces=True):
    """ Parses UKB encoding.ukb file.

    Args:
        insert_spaces: UKB format occasionally uses '+' instead of spaces. True to replace these, False to leave unchanged.
    Returns:
        list of tuples (encoding_id, encoding), where encoding is an instance of class Encoding
    """
    f.seek(0)
    encoding_text = f.read().strip()
    parse_tree = ukb_encoding_grammar.parse(encoding_text)
    return dict(UKBEncodingParser(insert_spaces).visit(parse_tree))


class Columns:
    def __init__(self, column_list):
        if isinstance(column_list, list):
            self._columns = {
                col['name']: col for _, col in column_list
            }
        elif isinstance(column_list, dict):
            self._columns = dict(**column_list)
        else:
            raise ValueError("column_list not recognised.")

    BASELINE_ASSESSMENT = 0
    REPEAT_ASSESSMENT = 1
    FIRST_IMAGING_ASSESSMENT = 2
    SECOND_IMAGING_ASSESSMENT = 3

    def get_column_idxs(self, title_or_field_id, flat=False):
        if isinstance(title_or_field_id, str):
            filter_fn = lambda col: col['title'].lower() == title_or_field_id.lower()
        else:
            filter_fn = lambda col: col['field_id'] == title_or_field_id

        columns = [(name, col) for name, col in self._columns.items() if filter_fn(col)]
        if flat:
            return np.array(list(zip(*columns))[0])

        max_instance_idx = reduce(
            lambda inst_num, col: max(inst_num, col[1]['instance_idx']),
            columns, 0
        )
        num_instances = max_instance_idx + 1

        return np.array([
            [name for name, col in columns if col['instance_idx'] == instance_idx]
            for instance_idx in range(num_instances)
        ])

    def get_field_id(self, title):
        try:
            return [col['field_id'] for _, col in self._columns.items()if col['title'].lower() == title.lower()][0]
        except:
            raise KeyError(f"No column with title '{title}'!")

    def get_column_info(self, column_name):
        return self._columns[column_name]

    def get_column_encoding(self, column_name):
        return self.get_column_info(column_name)['encoding']

    def get_titles(self):
        return set(col['title'] for col in self._columns.values())

    def get_all_of_type(self, t):
        return [name for name, col in self._columns.items() if col['type'] == t]

    def get_type_map(self, internal_mapping=None):
        def map_type(t):
            if internal_mapping and t in internal_mapping:
                return internal_mapping[t]
            return t
        return {name: map_type(col['type']) for name, col in self._columns.items()}

    def save(self, path_or_file):
        if isinstance(path_or_file, str):
            f = open(path_or_file, "wb")
            try:
                pickle.dump(self._columns, f)
            finally:
                f.close()
        else:
            pickle.dump(self._columns, path_or_file)

    @staticmethod
    def load(path_or_file) -> 'Columns':
        if isinstance(path_or_file, str):
            f = open(path_or_file, "rb")
            try:
                columns = pickle.load(f)
            finally:
                f.close()
        else:
            columns = pickle.load(path_or_file)
        return Columns(columns)

    @staticmethod
    def from_data(data_file, encodings) -> 'Columns':
        def column_process_fn(col_id, col_info):
            return {
                'encoding': encodings.get(col_info['eid'], Encoding.Unknown(col_info['eid'])),
                'field_id': col_info['fid'],
                'name': col_info['nam'],
                'title': col_info['title'],
                'instance_idx': col_info['ins'],
                'array_idx': col_info['arr'],
                'type': col_info['ing'],
                'column_id': col_id
            }
        return Columns(
            parse_ukb_data_header(data_file, column_process_fn=column_process_fn)
        )


    def __len__(self):
        return len(self._columns)




if __name__ == '__main__':
    encodings = UKBEncodingParser().visit(ukb_encoding_grammar.parse(test_example))
    print(encodings)

    codings = [{'ci': -1, 'mn': 'Do not know', 'pi': 0, 'se': 1, 'va': '-1'}, {'ci': -3, 'mn': 'Prefer not to answer', 'pi': 0, 'se': 1, 'va': '-3'}]
    e = Encoding(None, codings)
    print(e.decode(-1))
    print(e.decode(-3))
    print(e.decode("Value without corresponding code just passes through"))