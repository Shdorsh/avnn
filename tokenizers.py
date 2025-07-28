from torch import tensor, long
from json import dumps, load, JSONDecodeError
from os.path import isfile

class TagValueTokenizer:
    def __init__(self, row_delimiter="\n", column_separator="|", tag_value_separator=":"):
        self.vocab = None
        self.max_len = 0
        self.excluded_columns = []
        self.token_id = 0
        self.row_delimiter = row_delimiter
        self.column_separator = column_separator
        self.tag_value_separator = tag_value_separator

    def exclude_columns(self, exclusion_array):
        self.excluded_columns = exclusion_array

    def save_vocab(self):
        data = {
            'vocab': self.vocab,
            'max_len': self.max_len,
            'excluded_columns': self.excluded_columns
        }
        with open('vocab.json', "w") as fd:
            fd.write(dumps(data))

    def build_vocab(self, data_string):
        # Build vocabulary and determine global max length of tag-value pairs
        self.vocab = {}
        for row in data_string.strip().split(self.row_delimiter):
            for i, entry in enumerate(row.split(self.column_separator)):
                if i in self.excluded_columns or not entry:
                    continue
                parts = entry.split(self.tag_value_separator)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if part not in self.vocab:
                        self.vocab[part] = self.token_id
                        self.token_id += 1
        self.max_len = len(self.vocab)
        self.save_vocab()
        return self.max_len
    
    def get_features(self):
        return self.max_len

    def load_vocab(self, file):
        if not isfile(file):
            raise FileNotFoundError(f"The given file {file} could not be found!")
        with open(file) as fd:
            try:
                data = load(fd)
                self.vocab = data['vocab']
                self.max_len = data['max_len']
                self.excluded_columns = data.get('excluded_columns', [])
            except JSONDecodeError as e:
                raise ValueError(f"Failed to parse vocab JSON: {e}")

    def tokenize(self, data_string):
        # Ensure vocabulary and max_len are ready
        if self.vocab is None or self.max_len == 0:
            self.build_vocab(data_string)

        # Single row input expected: a string of tags separated by column_separator
        tokens = []
        for i, entry in enumerate(data_string.strip().split(self.column_separator)):
            if i in self.excluded_columns or not entry:
                continue
            tag_value = entry.split(self.tag_value_separator)
            tag = self.vocab.get(tag_value[0].strip(), 0)
            val = self.vocab.get(tag_value[1].strip(), 0) if len(tag_value) > 1 else 0
            tokens.append([tag, val])

        # Truncate if too long
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        # Pad if too short
        while len(tokens) < self.max_len:
            tokens.append([0, 0])

        return tensor(tokens, dtype=long)

__all__ = ['TagValueTokenizer']