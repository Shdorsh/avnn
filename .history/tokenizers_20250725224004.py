from torch import tensor, long
from json import dumps, load, JSONDecodeError
from os.path import isfile

class TagValueTokenizer:
    def __init__(self, row_delimiter="\n", column_separator="|", tag_value_separator=":"):
        self.vocab = None
        self.excluded_columns = []
        self.token_id = 0
        self.row_delimiter = row_delimiter
        self.column_separator = column_separator
        self.tag_value_separator = tag_value_separator

    def exclude_columns(self, exclusion_array):
        self.excluded_columns = exclusion_array

    def save_vocab(self):
        filetext = dumps(self.vocab)
        with open('vocab.json', "w") as fd:
            fd.write(filetext)

    def build_vocab(self, data_string):
        self.vocab = {}
        for row in data_string.strip().split(self.row_delimiter):
            i = -1
            for entry in row.split(self.column_separator):
                i += 1
                if i in self.excluded_columns:
                    continue
                for part in entry.split(self.tag_value_separator):
                    part = part.strip()
                    if part and part not in self.vocab:
                        self.vocab[part] = self.token_id
                        self.token_id += 1
        self.save_vocab()

    def load_vocab(self, file):
        if not isfile(file):
            raise FileNotFoundError(f"The given file {file} could not be found!")
        with open(file) as fd:
            try:
                self.vocab = load(fd)
            except Exception as e:
                raise ValueError(f"Failed to parse vocab JSON: {e}")

    def tokenize(self, data_string):
        if self.vocab is None:
            self.build_vocab(data_string)
        result = []
        max_len = 0

        # First pass: tokenize and find max row length
        for row in data_string.strip().split(self.row_delimiter):
            if row == '':
                continue
            row_data = []
            for i, entry in enumerate(row.split(self.column_separator)):
                if i in self.excluded_columns or not entry:
                    continue
                tag_value = entry.split(self.tag_value_separator)
                tag = self.vocab.get(tag_value[0].strip(), 0)
                val = self.vocab.get(tag_value[1].strip(), 0) if len(tag_value) > 1 else 0
                row_data.append([tag, val])
            max_len = max(max_len, len(row_data))
            result.append(row_data)

        # Pad rows to max_len
        for row_data in result:
            while len(row_data) < max_len:
                row_data.append([0, 0])

        return tensor(result, dtype=long)

__all__ = ['TagValueTokenizer']