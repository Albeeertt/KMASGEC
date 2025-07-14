import os
import json
import base64
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

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


def numpy_generator(filename: str):
    """Generador que lee JSON línea a línea y devuelve np.ndarrays."""
    with open(filename, 'r') as f:
        for line in f:
            sample = json.loads(line)
            # Decodificar X
            X_bytes = base64.b64decode(sample["X"]["data"])
            X = np.frombuffer(X_bytes, dtype=sample["X"]["dtype"])
            X = X.reshape(sample["X"]["shape"])
            # Decodificar Y
            Y_bytes = base64.b64decode(sample["Y"]["data"])
            Y = np.frombuffer(Y_bytes, dtype=sample["Y"]["dtype"])
            Y = Y.reshape(sample["Y"]["shape"])
            yield X, Y


class JsonIterableDataset(IterableDataset):
    """IterableDataset que utiliza el generador numpy_generator y convierte a torch.Tensor."""
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def __iter__(self):
        for X_np, Y_np in numpy_generator(self.filename):
            # Convertir a torch.Tensor y asegurarse dtype float32
            X_t = torch.from_numpy(X_np).float() # .unsqueeze(-1)  
            Y_t = torch.from_numpy(Y_np).float()
            yield X_t, Y_t


def get_pytorch_dataloader(
    filename: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
):
    """
    Crea un DataLoader de PyTorch para tu archivo JSON:
      - IterableDataset porque lee secuencialmente sin cargar todo en memoria.
      - shuffle: solo funciona si dataset implementa __len__; 
        para un shuffling más simple puedes usar buffer en memoria.
    """
    dataset = JsonIterableDataset(filename)

    # Si quieres shuffling real en IterableDataset, tendrías que implementarlo a mano
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and hasattr(dataset, "__len__"),
        num_workers=num_workers,
        pin_memory=True
    )
