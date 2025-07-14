import math
from multiprocessing import cpu_count
from typing import Dict, List
from pandas import DataFrame

def split_into_chunks(fasta: Dict[str, DataFrame], bed: Dict[int, DataFrame], n_cpus = None):
    n_cpus = n_cpus or (cpu_count() - 1)
    
    if len(bed) > 1:
        if len(bed) > n_cpus:
            n_chunks = np.ceil(len(bed) / n_cpus)
            contador = 1
            list_chunks_complete = []
            last_chunk_bed = {}
            for key in bed.keys():
                if n_chunks == 0:
                    last_chunk_bed[contador] = bed[key]
                    contador += 1
                else:
                    list_chunks_complete.append(({1: bed[key]}, fasta))
                    n_chunks -= 1
            if len(last_chunk_bed) > 0:
                list_chunks_complete.append(last_chunk_bed, fasta)
            return list_chunks_complete
                
        else:
            list_chunks_complete = [({1: bed[key]}, fasta) for key in bed.keys()]
            return list_chunks_complete
    else:
        chunks = np.array_split(bed[1], n_cpus)
        list_chunks_complete = [({1: chunk}, fasta) for chunk in chunks]
        return list_chunks_complete



def split_list_into_tables(list_records: List[Dict], limite, n_cpus = None):
    n_cpus = n_cpus or (cpu_count() - 1)
    return_list = [[] for _ in range(n_cpus)]
    len_list_records: int = len(list_records)
    value_quotient = len_list_records / limite
    how_for_cpu = math.floor(value_quotient / n_cpus)
    records_count: int = 0
    idx: int = 0
    if how_for_cpu < 1:
        while records_count < len_list_records:
            minim = idx*limite
            maxim = min((idx*limite)+limite, len_list_records)
            return_list[idx] = list_records[minim:maxim]
            idx += 1
            records_count += limite
    else:
        while records_count < len_list_records:
            minim = idx * (how_for_cpu * limite)
            if len(return_list)-1 == idx:
                maxim = len_list_records
                records_count = len_list_records
            else:     
                maxim = (idx * (how_for_cpu * limite)) + (how_for_cpu * limite)
                records_count += (how_for_cpu * limite)
            return_list[idx] = list_records[minim:maxim]
            idx += 1

    return return_list