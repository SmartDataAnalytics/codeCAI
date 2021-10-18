import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data, edge_path_transform):
        self.data = data
        self.edge_path_transform = edge_path_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        nl = self.data.iloc[index, 0]
        nl_char = self.data.iloc[index, 1]
        ast_seq = self.data.iloc[index, 2]
        edge_order_paths = self.data.iloc[index, 3]
        allowed_tokens = self.data.iloc[index, 4]

        if self.edge_path_transform:
            edge_order_paths = self.edge_path_transform(edge_order_paths)

        return nl, nl_char, ast_seq, edge_order_paths, allowed_tokens


class EdgePathTransform:
    def __init__(self, max_path_depth, path_multiple=0, device=None):
        self.max_path_depth = max_path_depth
        self.path_multiple = path_multiple
        self.device = device

    def __call__(self, paths):
        sentence_length = len(paths)
        paths_padded = torch.zeros(sentence_length, self.max_path_depth, dtype=torch.long, device=self.device)

        for i, path in enumerate(paths):
            path_len = len(path)
            path = [(path_len - 1 - ind) * self.path_multiple + x for ind, x in enumerate(path)]

            if path_len > self.max_path_depth:
                path = path[:self.max_path_depth]
                path_len = self.max_path_depth

            paths_padded[i, :path_len] = torch.as_tensor(path, dtype=torch.long, device=self.device)

        return paths_padded


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.has_id = ('id' in data.columns)
        self.has_ast_seq = ('ast_seq_list' in data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        nl = data[0]
        nl_char = data[1]
        snippet = data[2]
        idx = data[3] if self.has_id else None
        ast_seq = data[4] if self.has_ast_seq else None

        return nl, nl_char, snippet, idx, ast_seq
