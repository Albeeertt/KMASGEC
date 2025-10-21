import base64, orjson, numpy as np, tensorflow as tf
from collections import Counter

AUTOTUNE = tf.data.AUTOTUNE

def _scan_offsets(filename, min_len_seq, max_len_seq, limit=None):
    """Pasa una vez para quedarte solo con líneas válidas y respetar 'limit' por clase."""
    offsets, counts = [], Counter()
    off = 0
    with open(filename, "rb") as f:
        for raw in f:
            sample = orjson.loads(raw)
            L = int(sample["X"]["shape"][0])
            y = int(np.frombuffer(base64.b64decode(sample["Y"]["data"]),
                                  dtype=np.dtype(sample["Y"]["dtype"]))
                        .reshape(tuple(sample["Y"]["shape"])).ravel()[0])
            ok_len = (L >= min_len_seq[y]) and (L <= max_len_seq)
            ok_lim = True if not limit else (counts[y] < limit[y])
            if ok_len and ok_lim:
                offsets.append((off, len(raw)))
                counts[y] += 1
            off += len(raw)
    return offsets

def make_tf_dataset(filename: str,
                    min_len_seq: dict,
                    max_len_seq: int,
                    instance_generateDataset,
                    limit: dict | None = None,
                    kmer: bool = False,
                    pad_id: int = 0,
                    batch_size: int = 64,
                    shuffle: bool = True,
                    bucket_boundaries: list[int] | None = None,
                    bucket_batch_sizes: list[int] | None = None):
    """
    Devuelve tf.data.Dataset de (seq_padded[int32], label[int32], attention_mask[bool])
    """
    offsets = _scan_offsets(filename, min_len_seq, max_len_seq, limit)

    def gen():
        with open(filename, "rb") as f:
            for off, ln in offsets:
                f.seek(off)
                raw = f.read(ln)
                sample = orjson.loads(raw)

                x = np.frombuffer(base64.b64decode(sample["X"]["data"]),
                                  dtype=np.dtype(sample["X"]["dtype"])) \
                        .reshape(sample["X"]["shape"])
                y = int(np.frombuffer(base64.b64decode(sample["Y"]["data"]),
                                      dtype=np.dtype(sample["Y"]["dtype"]))
                            .reshape(sample["Y"]["shape"]).ravel()[0])

                if kmer:
                    x = instance_generateDataset.seq_to_kmer(x, sample["X"]["dtype"], y)
                else:
                    x = instance_generateDataset.seq_to_id(x, sample["X"]["dtype"], y)

                x = np.asarray(x).squeeze(-1).astype(np.int32)   # [L]
                yield x, np.int32(y)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),  # seq var-long
            tf.TensorSpec(shape=(),      dtype=tf.int32),  # label
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(offsets), 10000))

    # Bucketing opcional para longitudes parecidas
    if bucket_boundaries is not None:
        if bucket_batch_sizes is None:
            # mismo tamaño de batch para todos los buckets por defecto
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

        def element_length(seq, y):
            return tf.shape(seq)[0]

        ds = tf.data.experimental.bucket_by_sequence_length(
            element_length_func=element_length,
            bucket_boundaries=list(bucket_boundaries),
            bucket_batch_sizes=list(bucket_batch_sizes),
            padded_shapes=([None], []),
            padding_values=(tf.constant(pad_id, tf.int32),
                           tf.constant(0, tf.int32)),
            drop_remainder=False,
        )
    else:
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=([None], []),
            padding_values=(tf.constant(pad_id, tf.int32),
                           tf.constant(0, tf.int32))
        )

    # Añade attention_mask (True = token válido)
    ds = ds.map(lambda seq, y: (seq, y, tf.not_equal(seq, pad_id)),
                num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    return ds

