from itertools import product
import random
import bisect
import numpy as np

import logging
from typing import List, Dict
from pandas import DataFrame


class GenerateDataset:

    def __init__(self, string_seq: bool, agrupacion: int):
        self._logger = logging.getLogger(__name__)
        self._agrupacion = agrupacion
        logging.basicConfig(level=logging.INFO)
        self.dataset: List[Dict] = []
        if string_seq:
            bases = ['A','T','C','G']
        else:
            bases = ['0', '1', '2', '3']
        vocab = { ''.join(codon):idx for idx, codon in enumerate(product(bases, repeat=agrupacion)) }
        offset = len(vocab)  
        agrupacion -= 1
        while agrupacion != 0:
            for group in product(bases, repeat=agrupacion):
                vocab[''.join(group)] = offset
                offset += 1
            agrupacion -= 1

        self.vocabularyComplete = vocab

    def codon_to_ids(self, list_records: List[Dict], capar_c_inicio_fin: bool):
        list_ids: List[Dict] = []
        for record in list_records:
            seq = record['seq']
            type_seq = record['type']
            new_record = [self.vocabularyComplete[seq[i:i+3]] for i in range(0, len(seq), 3) ]
            if capar_c_inicio_fin:
                if type_seq == "CDS":
                    new_record = self.capar_cds_inicio_fin(new_record, True)
                else:
                    new_record = self.capar_cds_inicio_fin(new_record, False)
            list_ids.append({"type": type_seq, "seq": new_record})
        return list_ids

     
    def seq_to_id(self, seq_record: Dict, dtype, y):
        new_record_seq = []
        for i in range(0, len(seq_record), self._agrupacion):
            piece_of_seq = seq_record[i:i+self._agrupacion]
            piece_to_str = ''.join(piece_of_seq.astype(int).astype(str))
            new_record_seq.append(self.vocabularyComplete[piece_to_str])
        #new_record_seq = [self.vocabularyComplete[str(int(seq_record[i:i+self._agrupacion]))] for i in range(0, len(seq_record), self._agrupacion)]
        if y == 1: # Uno es el valor que obtiene el CDS.
            new_record_seq = new_record_seq[1:-1]
        return np.array(new_record_seq, dtype=dtype)

    def capar_cds_inicio_fin(self, record_seq: List[int], is_cds: bool):
        inicio = self.vocabularyComplete['ATG']
        fin = ( self.vocabularyComplete['TAA'], self.vocabularyComplete['TAG'], self.vocabularyComplete['TGA'])

        limit = len(self.vocabularyComplete)

        if is_cds:
            limit = 63

        if record_seq[0] == inicio:
            while True:
                num = random.randint(0, limit)
                if num != inicio:
                    break
                record_seq[0] = num
        if record_seq[-1] in fin:
            while True:
                num = random.randint(0, limit)
                if num not in fin:
                    break
            record_seq[-1] = num
        return record_seq

class GenerateNegativeData:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) 


    def _correctFormatGFF(self, gff: DataFrame):
        cds = gff[gff['type'] == "CDS"]
        return cds.to_dict(orient="records")


    
    def _parse_cds_intervals(self, list_records_cds: List[Dict]):
        """Devuelve dict {seqid: [(start, end), ...]} con todos los CDS del GFF3."""
        cds = {}
        for record in list_records_cds:
            seqid = record['chr']
            start, end = record['start'], record['end']
            cds.setdefault(seqid, []).append((start, end))

        # Merge intervals dentro de cada seqid
        merged = {}
        for seqid, ivs in cds.items():
            ivs.sort()
            m = []
            cur_s, cur_e = ivs[0]
            for s,e in ivs[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    m.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            m.append((cur_s, cur_e))
            merged[seqid] = m
        return merged

    def _overlap(self, intervals, s, e):
        # intervals es lista de (s_i,e_i) no solapados y ordenados
        # buscamos el primer interval con start > e
        i = bisect.bisect_right(intervals, (e, float('inf')))
        # el candidato es el anterior
        if i and intervals[i-1][1] >= s:
            return True
        return False


    def samples_negatives(self, fasta, gff, n, minL, maxL):
        list_cds_records = self._correctFormatGFF(gff)
        cds = self._parse_cds_intervals(list_cds_records)
        samples_negatives: List[Dict] = []
        while len(samples_negatives) < n:
            chr_selected = random.choice(list(cds.keys()))
            len_selected = random.randint(minL, maxL)
            max_len_chr = len(fasta[chr_selected])
            start_selected = random.randint(1, max_len_chr)
            end_selected = start_selected + len_selected
            if end_selected > max_len_chr:
                continue
            if not self._overlap(cds.get(chr_selected, []), start_selected, end_selected):
                contenido = fasta[chr_selected][start_selected:end_selected]
                contaminada: bool = not set(contenido).issubset({'A','T','C','G'})
                if not contaminada:
                    samples_negatives.append({"type": 'negative', "seq": contenido})
        return samples_negatives
