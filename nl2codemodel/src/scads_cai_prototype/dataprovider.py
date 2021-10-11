import pandas as pd


class DataProvider:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data: pd.DataFrame = None

    def provide_data(self, columns, optional_columns=['id', 'ast_seq_list'], column_default_values={}) -> pd.DataFrame:
        self.ensure_data_is_loaded()

        data = self.data

        for column in columns:
            if column not in self.data.columns:
                if column in column_default_values:
                    default_value = column_default_values[column]
                    kwargs = {column: default_value}
                    data = data.assign(**kwargs)
                else:
                    raise ValueError("Missing column in data: '%s'" % column)

        additional_columns = [col for col in optional_columns if col in data.columns and col not in columns]
        columns = columns + additional_columns
        return data[columns].copy(deep=False)

    def ensure_data_is_loaded(self):
        if self.data is None:
            self.data = self.load_data(self.data_path)

    def load_data(self, path):
        is_jsonl = path.lower().endswith('.jsonl')
        return pd.read_json(path, lines=is_jsonl, dtype=False)
