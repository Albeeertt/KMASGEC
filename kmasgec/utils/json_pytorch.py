import os
import json
import base64
import numpy as np
import torch
import orjson
from collections import Counter
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Dict

def merge_json(files : List[str], min_len_seq: int, max_len_seq: int, limit: Dict[int, int], output_path: str):
    with open(output_path, "w") as out:
        for fname in files:
            contador = Counter()
            with open(fname, "r") as inp:
                for line in inp:
                    sample = orjson.loads(line)
                    X_shape = int(sample["X"]["shape"][0])
                    Y_decoded = base64.b64decode(sample["Y"]["data"])
                    Y_arr = np.frombuffer(Y_decoded,
                        dtype=sample["Y"]["dtype"]
                        ).reshape(sample["Y"]["shape"]).copy()
                    Y_arr = int(Y_arr.flatten()[0])
                    if contador[Y_arr] < limit[Y_arr] and X_shape >= min_len_seq and X_shape <= max_len_seq:
                        out.write(line if line.endswith("\n") else (line + "\n"))
                        contador[Y_arr] += 1

def split_records_species(*species, max_len: int, overlap: int, route_out: str):
    def toBase64(array):
        array_bytes = array.tobytes()
        array_b64 = base64.b64encode(array_bytes).decode("utf-8")
        return {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "data": array_b64
        }
    list_route_out = []
    for specie in species:
        specie_name = (specie.split('/')[-1]).split('.')[0]
        route_specie_out: str = f'{route_out}{specie_name}_split.json'
        with open(specie,'r') as f:
            with open(route_specie_out, 'w') as out:
                for line in f:
                    sample = orjson.loads(line)
                    X_shape = int(sample['X']['shape'][0])
                    Y_decoded = base64.b64decode(sample["Y"]["data"])
                    Y_arr = np.frombuffer(Y_decoded,
                            dtype=sample["Y"]["dtype"]
                    ).reshape(sample["Y"]["shape"]).copy()
                    Y_arr = int(Y_arr.flatten()[0])
                    # TODO: borrar esta parte
                    start_decoded = base64.b64decode(sample['START']['data'])
                    start_rec = np.frombuffer(start_decoded, dtype=sample['START']['dtype']).reshape(sample['START']['shape']).copy()
                    end_decoded = base64.b64decode(sample['END']['data'])
                    end_rec = np.frombuffer(end_decoded, dtype=sample['END']['dtype']).reshape(sample['END']['shape']).copy()
                    if Y_arr == 2:
                        continue
                    if X_shape > max_len:
                        X_decoded = base64.b64decode(sample['X']['data'])
                        X_full = np.frombuffer(X_decoded, dtype=np.dtype(sample['X']['dtype'])).reshape(sample["X"]["shape"]).copy()
                        for start in range(0, (X_shape - max_len + 1), (max_len-overlap)):
                            end = start + max_len
                            x_chunk = X_full[start:end]
                            new_sample = dict(sample)
                            new_sample["X"] = toBase64(x_chunk)
                            new_sample["START"] = toBase64(start_rec+start)
                            new_sample['END'] = toBase64(start_rec+end)
                            json.dump(new_sample, out)
                            out.write("\n")
                        if ((X_shape - max_len) % (max_len - overlap))  != 0:
                            start = end
                            end = X_shape
                            x_chunk = X_full[start:end]
                            new_sample = dict(sample)
                            new_sample["X"] = toBase64(x_chunk)
                            new_sample['START'] = toBase64(start_rec+start)
                            new_sample['END'] = toBase64(end_rec)
                            json.dump(new_sample, out)
                            out.write("\n")
                    else:
                        out.write(line if line.endswith("\n") else (line + "\n"))
        list_route_out.append(route_specie_out)
    return route_specie_out

def save_all_to_json(*chunks , filename, names):

    def toBase64(array):
        array_bytes = array.tobytes()
        array_b64 = base64.b64encode(array_bytes).decode("utf-8")
        return {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "data": array_b64
        }
    mode = "w" if not os.path.exists(filename) else 'a'
    with open(filename, mode) as file:
        for row in zip(*chunks):
            json.dump({key: toBase64(value) for key, value in zip(names, row)}, file)
            file.write("\n")


def save_chunks_to_json_split_maxLen(X_chunk, y_chunk, filename: str, maxLen_seq: int):
    def toBase64(array):
        array_bytes = array.tobytes()
        array_b64   = base64.b64encode(array_bytes).decode("utf-8")
        return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "data":  array_b64
        }
    identify: int = 0
    mode = 'w' if not os.path.exists(filename) else 'a'
    with open(filename, mode) as file:
        for x_table, y_table in zip(X_chunk, y_chunk):
            if x_table.shape[0] > maxLen_seq:
                x_table_parts = np.split(x_table, maxLen_seq)
                for part in x_table_parts:
                    json.dump({'X': toBase64(part), 'Y': toBase64(y_table), 'Identify': identify}, file)
            else:
                json.dump({'X': toBase64(x_table), 'Y': toBase64(y_table)}, file)   

def save_chunks_to_json(X_chunk, y_chunk, filename):
    """Guarda un ndarray en JSON de forma eficiente usando base64."""
    def toBase64(array):
        array_bytes = array.tobytes()
        array_b64   = base64.b64encode(array_bytes).decode("utf-8")
        return {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "data":  array_b64
        }

    mode = 'w' if not os.path.exists(filename) else 'a'
    with open(filename, mode) as file:
        for x_table, y_table in zip(X_chunk, y_chunk):
            json.dump({"X": toBase64(x_table), "Y": toBase64(y_table)}, file)
            file.write("\n")


