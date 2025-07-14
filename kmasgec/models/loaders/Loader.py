import numpy as np
import base64
import orjson
import torch
from torch.utils.data import IterableDataset
import itertools
import re
from collections import Counter

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from typing import Dict

class Base64JSONIterableDataset(Dataset):
    """
    Lee línea a línea tu JSON en base64 y devuelve tuplas
    (features, labels) como torch.Tensor.
    """
    def __init__(self, filename: str, max_len_seq:int, instance_generateDataset, limit: Dict[str, int] = None):
        super().__init__()
        self.filename = filename
        self._max_len_seq = max_len_seq
        self._instance_generateDataset = instance_generateDataset
        self._f = None

        self.counter = Counter()
        self.counter_binary = Counter()
        self.offsets = []
        offset = 0
        shape_pat = re.compile(rb'"shape"\s*:\s*\[\s*(\d+)')
        y_pat = re.compile(
            rb'"Y"\s*:\s*\{\s*'
            rb'"shape"\s*:\s*\[\s*([0-9]+)\s*\]\s*,\s*'    # (1) primer elemento de shape
            rb'"dtype"\s*:\s*"([^"]+)"\s*,\s*'             # (2) dtype como string
            rb'"data"\s*:\s*"([A-Za-z0-9+/=]+)"'           # (3) blob base64
            rb'\s*\}'
        )
        with open(self.filename, 'rb') as f:
            for raw in f:
                length = len(raw)
                if not raw.rstrip(b'\n').endswith(b'}'):
                    offset += length
                    continue
                m = shape_pat.search(raw)
                if m and int(m.group(1)) <= max_len_seq:
                        
                        sample = orjson.loads(raw.decode('utf-8'))
                        y_meta = sample['Y']
                        shape = tuple(y_meta["shape"])
                        dtype = np.dtype(y_meta["dtype"])
                        b64   = y_meta["data"]
    
                        y_bytes = base64.b64decode(b64)
                        y_arr   = np.frombuffer(y_bytes, dtype=dtype).reshape(shape)
                        y_val   = int(y_arr.flatten()[0])
                        if limit: 
                            if self.counter[y_val] >= limit[y_val]:
                                offset += length
                                continue
                        
                        self.offsets.append((offset, len(raw)))
                        self.counter[y_val] += 1
                        if y_val == -1:
                            self.counter_binary[0] += 1
                        else:
                            self.counter_binary[1] += 1
                offset += len(raw)
            

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if idx >= len(self.offsets):
            raise IndexError
        if self._f is None:
            worker = get_worker_info()
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
        Y_arr = int(Y_arr.flatten()[0])
                # Convierte a torch.Tensor
                # Ajusta el dtype según tu tarea (float / long / etc.)
        X_arr = self._instance_generateDataset.seq_to_id(X_arr, sample["X"]["dtype"], Y_arr)
        #X_arr = X_arr.copy()
        X_tensor = torch.from_numpy(X_arr).long()
        X_tensor = X_tensor.squeeze(-1)
        Y_tensor = torch.tensor(Y_arr, dtype=torch.long) 

        return X_tensor, Y_tensor


class MyDataset(Dataset):
    def __init__(self, data):
        self.data     = data
        self.type2idx = {'CDS': 1, 'negative': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        t = self.type2idx[d["type"]]            
        s = torch.tensor(d["seq"], dtype=torch.long)
        return t, s


def collate_fn(batch, padding_value: int):
    """
    batch: lista de (t, s) para i en la muestra del DataLoader
    """
    seqs, types = zip(*batch)
    types = torch.stack(types).long()      # [B]
    types2 = types.ne(-1).long()
    # pad_sequence devuelve [L_max, B], lo transponemos a [B, L_max]
    seqs  = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
    # si tu modelo necesita máscara:
    mask  = (seqs != padding_value)
    return types, types2, seqs, mask


def collate_fn_oneHead(batch, padding_value: int):
    seqs, types = zip(*batch)
    types_list = [3 if t.item() == -1 else t.item() for t in types]
    types = torch.tensor(types_list, dtype=torch.long)
    seqs  = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
    mask  = (seqs != padding_value)
    return types, seqs, mask




class Borrar(Dataset):
    """
    Lee línea a línea tu JSON en base64 y devuelve tuplas
    (features, labels) como torch.Tensor.
    """
    def __init__(self, filename: str, max_len_seq:int, instance_generateDataset, limit: Dict[str, int] = None):
        super().__init__()
        self.filename = filename
        self._max_len_seq = max_len_seq
        self._instance_generateDataset = instance_generateDataset
        self._f = None

        self.counter = Counter()
        self.counter_binary = Counter()
        self.counter_no = Counter()
        self.offsets = []
        contador_total = 0
        contador_no = 0
        offset = 0
        shape_pat = re.compile(rb'"shape"\s*:\s*\[\s*(\d+)')
        y_pat = re.compile(
            rb'"Y"\s*:\s*\{\s*'
            rb'"shape"\s*:\s*\[\s*([0-9]+)\s*\]\s*,\s*'    # (1) primer elemento de shape
            rb'"dtype"\s*:\s*"([^"]+)"\s*,\s*'             # (2) dtype como string
            rb'"data"\s*:\s*"([A-Za-z0-9+/=]+)"'           # (3) blob base64
            rb'\s*\}'
        )
        with open(self.filename, 'rb') as f:
            for raw in f:
                length = len(raw)
                if not raw.rstrip(b'\n').endswith(b'}'):
                    offset += length
                    continue
                m = shape_pat.search(raw)
                if m and int(m.group(1)) <= max_len_seq:
                        contador_total += 1
                        sample = orjson.loads(raw.decode('utf-8'))
                        y_meta = sample['Y']
                        shape = tuple(y_meta["shape"])
                        dtype = np.dtype(y_meta["dtype"])
                        b64   = y_meta["data"]

                        y_bytes = base64.b64decode(b64)
                        y_arr   = np.frombuffer(y_bytes, dtype=dtype).reshape(shape)
                        y_val   = int(y_arr.flatten()[0])
                        if limit:
                            if self.counter[y_val] >= limit[y_val]:
                                offset += length
                                continue

                        self.offsets.append((offset, len(raw)))
                        self.counter[y_val] += 1
                        if y_val == -1:
                            self.counter_binary[0] += 1
                        else:
                            self.counter_binary[1] += 1
                elif m and int(m.group(1)) > max_len_seq:
                    contador_total += 1
                    contador_no += 1
                    sample = orjson.loads(raw.decode('utf-8'))
                    y_meta = sample['Y']
                    shape = tuple(y_meta["shape"])
                    dtype = np.dtype(y_meta["dtype"])
                    b64   = y_meta["data"]

                    y_bytes = base64.b64decode(b64)
                    y_arr   = np.frombuffer(y_bytes, dtype=dtype).reshape(shape)
                    y_val   = int(y_arr.flatten()[0])
                    self.counter_no[y_val] += 1
                elif not m:
                    print("Espera ¿qué cojones?")
                offset += len(raw)
        self.porcentaje_no = (contador_no * 100)/contador_total


    def __len__(self):
        return len(self.offsets)


    def __getitem__(self, idx):
        if idx >= len(self.offsets):
            raise IndexError
        if self._f is None:
            worker = get_worker_info()
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

                # Convierte a torch.Tensor
                # Ajusta el dtype según tu tarea (float / long / etc.)
        X_arr = self._instance_generateDataset.seq_to_id(X_arr, sample["X"]["dtype"])
        #X_arr = X_arr.copy()
        X_tensor = torch.from_numpy(X_arr).long()
        X_tensor = X_tensor.squeeze(-1)
        Y_arr = Y_arr.flatten()[0]
        Y_tensor = torch.tensor(int(Y_arr), dtype=torch.long)

        return X_tensor, Y_tensor


