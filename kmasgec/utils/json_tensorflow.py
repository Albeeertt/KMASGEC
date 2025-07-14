
from typing import Generator, Tuple

import base64
import os
import tensorflow as tf
import numpy as np
import json


def save_chunks_to_json(X_chunk, y_chunk, filename):
    """Guarda un ndarray en JSON de forma eficiente usando base64."""
    def toBase64(array):

        array_bytes = array.tobytes()  # Convierte a binario
        array_b64 = base64.b64encode(array_bytes).decode("utf-8")  # Codifica en base64
        
        return {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "data": array_b64
        }

    mode = 'w' if not os.path.exists(filename) else 'a'

    with open(filename, mode) as file:
        for x_table, y_table in zip(X_chunk, y_chunk):
            json.dump({"X": toBase64(x_table), "Y": toBase64(y_table)}, file)
            file.write("\n")



def data_generator(filename: str) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generador que lee datos desde un JSON codificado en base64 y los devuelve como muestras individuales."""
    
    with open(filename, 'r') as file:
        for line in file:
            sample = json.loads(line)  # Deserializar la línea de JSON.
            # Decodificar el array X y Y desde base64
            X_decoded = base64.b64decode(sample["X"]["data"])
            Y_decoded = base64.b64decode(sample["Y"]["data"])
            
            # Reconstruir los arrays a partir de los datos decodificados
            X_array = np.frombuffer(X_decoded, dtype=sample["X"]["dtype"]).reshape(sample["X"]["shape"])
            Y_array = np.frombuffer(Y_decoded, dtype=sample["Y"]["dtype"]).reshape(sample["Y"]["shape"])
            
            yield X_array, Y_array 

def get_tf_dataset(filename: str, k: int, columns: int, batch_size: int =32, buffer_shuffle: int = 124 ):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(filename),
        output_signature=(
            tf.TensorSpec(shape=(4**k, columns, 1), dtype=tf.float32),      
            tf.TensorSpec(shape=(4**k, columns), dtype=tf.float32)      
        )
    )
    
    return dataset.shuffle(buffer_size=buffer_shuffle).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)