import pandas as pd
import os
import numpy as np
import logging
from Bio import SeqIO
import pyranges as pr
from collections import Counter

from typing import Dict, List
from pandas import DataFrame


class CleanData:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.dataset: List[Dict] = []


    def obtain_gff(self, route: str, encoding: str = 'utf-8') -> DataFrame:
        '''Devuelve todos los cromosomas de la especie junto a su fichero GFF3 como un dataframe'''
        data = pd.read_csv(route, comment='#', sep='\t', header=None, encoding= encoding)
        data.columns = ['chr','db','type','start','end','score','strand','phase','attributes']
        data['old_idx'] = data.index
        return data
    
    def obtain_dicc_fasta(self, route: str, mapping = None) -> Dict[str, str]:
        '''Devuelve todos los cromosomas de la especie junto a su fichero fasta como un string'''
        all_fasta : Dict[str,str] = {}
        with open(route, 'r') as file:
            for record in SeqIO.parse(file, "fasta"):
                if mapping:
                    all_fasta[mapping[record.id]] = str(record.seq).upper()
                else:
                    all_fasta[record.id] = str(record.seq).upper()
        return all_fasta


    def select_elements_gff(self, selected : List[str], gff: DataFrame, check: bool = False) -> DataFrame:    
        '''Selecciona del los dataframes de los archivos GFF3 las clases deseadas.'''
        mask = gff['type'].isin(selected)
        clean_gff : DataFrame = gff[mask]

        if check:
            dict_types_count_original = gff.value_counts().to_dict()
            dict_types_count_new = clean_gff.value_counts().to_dict()
            for shared_key in dict_types_count_new:
                assert dict_types_count_new[shared_key] == dict_types_count_original[shared_key]

        return clean_gff



    def extract_cds(self, gff: DataFrame, fasta: Dict[str, str], check: bool = False) -> List[Dict]:
        def complement(seq : str):
            complement = {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G': 'C'
            }
            complementaria : str = ''.join([complement.get(nucleotide, 'N') for nucleotide in seq]) # complementaria
            return complementaria[::-1] # invertida

        problemas = []
        removed_samples = Counter()

        gff['Parent'] = gff['attributes'].str.extract(r'Parent=([^;]+)', expand=False)
        gff_dropna = gff.dropna(subset=['Parent'])
        removed_samples['Eliminated by NA in column Parent'] = gff.shape[0] - gff_dropna.shape[0]
        bed_file = gff_dropna.sort_values(['Parent','start'], ascending=[True, True])

        dataset_dict : Dict[str, Dict] = {}
        dataset_old_idx : Dict[str, Dict] = {}
        list_bed : List = bed_file.to_dict(orient='records')
        for record in list_bed:
            fasta_a_usar : str = str(record['chr'])
            if fasta_a_usar not in list(fasta.keys()):
                problemas.append(fasta_a_usar)
                removed_samples['Eliminated by fasta not found'] += 1
                continue
            
            fasta_file : str = fasta[fasta_a_usar]
            if dataset_dict.get(fasta_a_usar, -1) == -1:
                dataset_dict[fasta_a_usar] = {}
                dataset_old_idx[fasta_a_usar] = {}

            if dataset_dict[fasta_a_usar].get(record["Parent"], -1) != -1:
                removed_samples['Eliminated by overlap'] += 1

            if record['start'] > record['end']:
                removed_samples['Eliminated by start sequence is bigger than end sequence'] += 1
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.'):
                    dataset_dict[fasta_a_usar][record["Parent"]] = dataset_dict[fasta_a_usar].get(record["Parent"], "") + fasta_file[record['start']-1:record['end']]
                    dataset_old_idx[fasta_a_usar][record["Parent"]] = dataset_old_idx[fasta_a_usar].get(record["Parent"], []) + [record['old_idx']]
            elif record['strand'] == '-':
                    dataset_dict[fasta_a_usar][record["Parent"]] = complement(fasta_file[record['start']-1:record['end']]) + dataset_dict[fasta_a_usar].get(record["Parent"], "")
                    dataset_old_idx[fasta_a_usar][record["Parent"]] = dataset_old_idx[fasta_a_usar].get(record["Parent"], []) + [record['old_idx']]


        final_dataset: List[Dict] = [
            {"type": "CDS", "seq":seq, "old_idx": dataset_old_idx[key][parent_key]}
            for key, inner_dict in dataset_dict.items()
            for parent_key, seq       in inner_dict.items()
        ]


        if check:
            eliminated_samples: int = 0
            for value in removed_samples.values():
                eliminated_samples += value
            assert gff.shape[0] == len(final_dataset) + eliminated_samples
            overlap_samples = 0
            for record in final_dataset:
                overlap_samples += len(record['old_idx'])
            assert gff.shape[0] == overlap_samples
            self._logger.info("Muestras eliminadas en extract_cds: ")
            self._logger.info(removed_samples)

        self.dataset = final_dataset
        return final_dataset
    

    def clean_cds(self, list_records: List[Dict], check: bool = False):
        new_list_record: List[Dict] = []
        contador = 0
        for record in list_records:
            if 'ATG' != record['seq'][:3] or record['seq'][-3:] not in ("TAA", "TAG", "TGA") or len(record['seq']) % 3 != 0:
                contador += 1
            else:
                new_list_record.append(record)

        if check:
            assert len(list_records) == len(new_list_record)+contador
            self._logger.info("Muestras eliminadas en clean_cds: ")
            self._logger.info(contador)
        
        return new_list_record

    def extract_sequences_mRNA(self, gff: DataFrame, fasta: Dict[str, str], check: bool = False) -> List[Dict]:
        '''Extrae las secuencias del archivo fasta mediante el archivo GFF3 (donde están todos los cromosomas).
        Sigue la misma lógica que Bedtools. Añadir un nucleótido de más al final.
        Elementos con estructura {'seq': ... , 'type': ...}'''
        def complement(seq : str):
            complement = {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G': 'C'
            }
            complementaria : str = ''.join([complement.get(nucleotide, 'N') for nucleotide in seq]) # complementaria
            return complementaria[::-1] # invertida


        problemas = []
        minimo: int = 10
        maximo: int = 500000
        final_dataset : List[Dict] = []
        remove_samples = Counter()

        gff['Parent'] = gff['attributes'].str.extract(r'Parent=([^;]+)', expand=False)
        gff_dropParent = gff.dropna(subset=['Parent'])
        remove_samples['Eliminated by NA in column Parent'] = gff.shape[0] - gff_dropParent.shape[0]
        gff_dropParent['ID'] = gff_dropParent['attributes'].str.extract(r'ID=([^;]+)', expand=False)
        gff_dropId = gff_dropParent.dropna(subset=['ID'])
        remove_samples['Eliminated by NA in column ID'] = gff_dropParent.shape[0] - gff_dropId.shape[0]
        bed_file = gff_dropId.sort_values(['Parent','start'], ascending=[True, True])
        list_bed : List = bed_file.to_dict(orient='records')
        parent_type: Dict[str, str] = {}
        for record in list_bed:
            if record['type'] == "mRNA":
                parent_type[record['Parent']] = "gene"
            parent_type[record['ID']] = record['type']
        for record in list_bed:
            if record['type'] == 'mRNA':
                remove_samples['Eliminated by be mRNA'] += 1
                continue
            if record['type'] == 'intron' and parent_type.get(record['Parent'], -1) == -1:
                remove_samples['Eliminated by be intron but not in mRNA'] += 1
                continue
            elif parent_type.get(record['Parent'], -1) == -1 or parent_type[record['Parent']] != "mRNA":
                remove_samples['Eliminated by not in mRNA'] += 1
                continue
            fasta_a_usar : str = str(record['chr'])
            if fasta_a_usar not in list(fasta.keys()):
                remove_samples['Eliminated by not fasta found'] += 1
                problemas.append(fasta_a_usar)
                continue
            fasta_file : str = fasta[fasta_a_usar]
            longitud = (int(record['end']) - int(record['start'])-1)
            if (longitud < minimo) or (longitud > maximo):
                remove_samples['Eliminated by less than minimun or more than maximum'] += 1
                continue
            if record['start'] > record['end']:
                remove_samples['Eliminated by start sequence bigger than end sequence'] += 1
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.'):
                final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type'], 'old_idx': record['old_idx']})
            elif record['strand'] == '-':
                final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type'], 'old_idx': record['old_idx']})

        if check:
            samples_eliminated = 0
            for value in remove_samples.values():
                samples_eliminated += value
            assert gff.shape[0] == len(final_dataset)+samples_eliminated
            self._logger.info("Muestas eliminadas en extract_sequences_mRNA: ")
            self._logger.info(remove_samples)

        
        self.dataset = final_dataset
        return final_dataset

    
    def extract_sequences_counting_chr(self, gff:  DataFrame, fasta: Dict[str, str], check: bool = False) -> List[Dict]:
        '''Extrae las secuencias del archivo fasta mediante el archivo GFF3 (donde están todos los cromosomas).
        Sigue la misma lógica que Bedtools. Añadir un nucleótido de más al final.
        Elementos con estructura {'seq': ... , 'type': ...}'''
        def complement(seq : str):
            complement = {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G': 'C'
            }
            complementaria : str = ''.join([complement.get(nucleotide, 'N') for nucleotide in seq]) # complementaria
            return complementaria[::-1] # invertida

        problemas = []
        minimo: int = 10
        maximo: int = 500000
        final_dataset : List[Dict] = []
        remove_samples = Counter()

        list_bed : List = gff.to_dict(orient='records')
        for record in list_bed:
            fasta_a_usar : str = str(record['chr'])
            if fasta_a_usar not in list(fasta.keys()):
                remove_samples['Eliminated by fasta not found'] += 1
                problemas.append(fasta_a_usar)
                continue
            fasta_file : str = fasta[fasta_a_usar]
            longitud = (int(record['end']) - int(record['start'])-1)
            if (longitud < minimo) or (longitud > maximo):
                remove_samples['Eliminated by less than minimun or more than maximum'] += 1
                continue
            if record['start'] > record['end']:
                remove_samples['Eliminated by start sequence bigger than end sequence'] += 1
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.'):
                final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type'], 'old_idx': record['old_idx']})
            elif record['strand'] == '-':
                final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type'], 'old_idx': record['old_idx']})

        if check:
            samples_eliminated = 0
            for value in remove_samples.values():
                samples_eliminated += value
            assert gff.shape[0] == len(final_dataset)+samples_eliminated
            self._logger.info("Muestras eliminadas en extract_sequences_counting_chr: ")
            self._logger.info(remove_samples)
        
        self.dataset = final_dataset
        return final_dataset

    def remove_sample_contaminated(self, dataset : List[Dict], check: bool = False) -> List[Dict]:
        '''Elimina las muestras contaminadas, es decir, la que no contienen el nucleótido A, C, T o G.'''

        clean_final_dataset : List = []
        count_contaminated = 0

        for record in dataset:
            contaminada: bool = not set(record['seq']).issubset({'A','T','C','G'})
            if not contaminada:
                clean_final_dataset.append(record)
            else:
                count_contaminated += 1

        if check:
            assert len(dataset) == len(clean_final_dataset) + count_contaminated
            self._logger.info("Muestras eliminadas en remove_sample_contaminated: ")
            self._logger.info(count_contaminated)
        
        self.dataset = clean_final_dataset
        return clean_final_dataset
