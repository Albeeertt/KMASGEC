import numpy as np
import base64
import orjson
import torch
from torch.utils.data import IterableDataset
import itertools
import re
from collections import Counter
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from typing import Dict

class Base64JSONIterableDataset(Dataset):
    """
    Lee línea a línea tu JSON en base64 y devuelve tuplas
    (features, labels) como torch.Tensor.
    """
    def __init__(self, filename: str, min_len_seq: Dict[int, int],  max_len_seq:int, instance_generateDataset, limit: Dict[int, int] = None, kmer: bool = False):
        super().__init__()
        self.filename = filename
        self._min_len_seq = min_len_seq
        self._max_len_seq = max_len_seq
        self._instance_generateDataset = instance_generateDataset
        self._kmer = kmer
        self._k_fold = []

        self.counter = Counter()
        offset = 0
        self.offsets = []

        with open(self.filename, 'rb') as f:
            for raw in f:
                sample = orjson.loads(raw)
                X_shape = int(sample["X"]["shape"][0])
                Y_decoded = base64.b64decode(sample["Y"]["data"])
                Y_arr = np.frombuffer(Y_decoded,
                        dtype=np.dtype(sample["Y"]["dtype"])
                        ).reshape(tuple(sample["Y"]["shape"]))
                Y_arr = int(Y_arr.flatten()[0])

                if X_shape >= min_len_seq[Y_arr] and X_shape <= max_len_seq:
                        
                    if limit: 
                        if self.counter[Y_arr] >= limit[Y_arr]:
                            offset += len(raw)
                            continue
                        
                    self.offsets.append((offset, len(raw)))
                    self.counter[Y_arr] += 1
                    self._k_fold.append(Y_arr)
                offset += len(raw)
            

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if idx >= len(self.offsets):
            raise IndexError
        self._f = open(self.filename, 'rb')

        self._f.seek(self.offsets[idx][0])
        raw = self._f.read(self.offsets[idx][1])
        line = raw.decode('utf-8').strip()
        sample = orjson.loads(line)
                # Decodifica X
        X_decoded = base64.b64decode(sample["X"]["data"])
        X_arr = np.frombuffer(X_decoded,
                dtype=sample["X"]["dtype"]
        ).reshape(sample["X"]["shape"])
                # Decodifica Y
        Y_decoded = base64.b64decode(sample["Y"]["data"])
        Y_arr = np.frombuffer(Y_decoded,
            dtype=sample["Y"]["dtype"]
        ).reshape(sample["Y"]["shape"]).copy()
        # Y_arr = int(Y_arr.flatten()[0])
        if self._kmer:
            X_arr = self._instance_generateDataset.seq_to_kmer(X_arr, sample["X"]["dtype"], Y_arr)
        else:
            X_arr = self._instance_generateDataset.seq_to_id(X_arr, sample["X"]["dtype"], Y_arr)
        X_tensor = torch.from_numpy(X_arr).long()
        X_tensor = X_tensor.squeeze(-1)
        Y_tensor = torch.tensor(Y_arr, dtype=torch.long) 

        place = base64.b64decode(sample["Place"]["data"])
        place = np.frombuffer(place,
            dtype=sample["Place"]["dtype"]
        ).reshape(sample["Place"]["shape"]).copy()
        place = int(place.flatten()[0])

        

        return X_tensor, Y_tensor, place


def generate_attn_mask(query: torch.Tensor, key: torch.Tensor, padding_value: int, num_heads: int):
    
    query_mask = (query != padding_value)
    key_mask = (key != padding_value)
    attn_mask = query_mask.unsqueeze(2) | key_mask.unsqueeze(1)
    attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
    attn_mask = attn_mask.bool()
    return attn_mask

def generate_key_padding_mask(key: torch.Tensor, padding_value: int):
    key_mask = (key != padding_value)
    key_mask = key_mask.bool()
    return key_mask

def collate_fn_oneHead(batch, padding_value: int):
    seqs, types, places = zip(*batch)

    types = torch.tensor(types, dtype=torch.long)
    # types = torch.stack([torch.tensor(t, dtype=torch.float32) for t in types])
    seqs  = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
    mask  = generate_key_padding_mask(seqs, padding_value)
    return seqs, types, mask, places
