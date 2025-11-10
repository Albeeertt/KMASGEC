import pandas as pd
import os
import numpy as np
import logging
from Bio import SeqIO
import pyranges as pr
from collections import Counter

from typing import Dict, List
from pandas import DataFrame

import random


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
        '''
        Devuelve todos los cromosomas de la especie junto a su fichero fasta como un string.
        Mapping nos sirve para modificar los keys del diccionario del fasta, es decir, los identificadores de los chr/scaffols por otros.
        '''
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
        Elementos con estructura {'seq': ... , 'type': ..., 'old_idx': ...}'''
        def complement(seq : str):
            complement = {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G': 'C'
            }
            complementaria : str = ''.join([complement.get(nucleotide, 'N') for nucleotide in seq]) # complementaria
            return complementaria[::-1] # invertida

        problemas_chr = []
        final_dataset : List[Dict] = []
        remove_samples = []

        list_bed : List = gff.to_dict(orient='records')
        for record in list_bed:
            fasta_a_usar : str = str(record['chr'])
            if fasta_a_usar not in list(fasta.keys()):
                problemas_chr.append(record['old_idx'])
                continue
            fasta_file : str = fasta[fasta_a_usar]
            if record['start'] > record['end']:
                remove_samples.append(record['old_idx'])
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.'):
                final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type'], 'old_idx': record['old_idx'], 'proportions': record['proportions'], 'COMPLETENESS': record['COMPLETENESS']})
            elif record['strand'] == '-':
                final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type'], 'old_idx': record['old_idx'], 'proportions': record['proportions'], 'COMPLETENESS': record['COMPLETENESS']})

        if check:
            samples_eliminated = 0
            for value in remove_samples.values():
                samples_eliminated += value
            assert gff.shape[0] == len(final_dataset)+samples_eliminated
            self._logger.info("Muestras eliminadas en extract_sequences_counting_chr: ")
            self._logger.info(remove_samples)
        
        self.dataset = final_dataset
        return final_dataset, problemas_chr, remove_samples

    def obtain_gene_w_mRNA(self, dataset: DataFrame, keep_classes: List[str], attr_split: bool = False, check: bool = False):
        '''
        1. Obtiene los genes que poseen mRNA y elimina el resto de genes. Solo elimina los genes que dan lugar a mRNA, el otro tipo de muestras las almacena.
        Una parte muy importante de esta función es que mantiene la estructura interna del gen que da lugar al mRNA, por tanto, se mantienen clases como exon, CDS, UTR, intrón, etc.
        '''

        list_records: List[Dict] = dataset.to_dict(orient="records")

        if not attr_split:
            new_list_records = []
            for record in list_records:
                record['ID'] = dict( part.split("=", 1) for part in record['attributes'].split(";")).get("ID", 'None')  
                record['Parent'] = dict( part.split("=", 1) for part in record['attributes'].split(";")).get("Parent", 'None') 
                new_list_records.append(record)
            list_records = new_list_records


        # 1.1 elements to keep
        records_to_keep: List[Dict] = [ record for record in list_records if record['type'] in keep_classes]


        # 1.2 filters the gene that don't produce mRNA.
        # 1.2.1 Obtain ids and conexions.
        dict_ids_record: Dict = {}
        gene_mRNA_record: Dict = {}
        for record in list_records:
            dict_ids_record[record['ID']] = record
            if record['type'] == 'mRNA':
                gene_mRNA_record[record['Parent']] = record['ID']

        # 1.2.2 Obtain records belong to mRNA/gene.
        records_genes_produce_mRNA = []
        remove_elements = []
        for record in list_records:
            if dict_ids_record.get(record['Parent'], -1) != -1 and dict_ids_record[record['Parent']]['type'] == 'mRNA':
                records_genes_produce_mRNA.append(record)
            elif record['type'] == "mRNA":
                records_genes_produce_mRNA.append(record)
            elif record['type'] == "gene" and gene_mRNA_record.get(record['ID'], -1) != -1:
                records_genes_produce_mRNA.append(record)
            elif dict_ids_record.get(record['Parent'], -1) != -1 and dict_ids_record[record['Parent']]['type'] == 'gene' and gene_mRNA_record.get(record['Parent'], -1) != -1:
                records_genes_produce_mRNA.append(record)
            else:
                remove_elements.append(record['old_idx'])

        if check:
            check_list_records = [record for record in list_records if record['type'] in ('gene', 'mRNA', 'exon', 'intron', 'CDS', 'three_prime_UTR', 'five_prime_UTR', 'UTR')]
            assert len(check_list_records) == len(records_genes_produce_mRNA)

        records_genes_produce_mRNA.extend(records_to_keep)

        return records_genes_produce_mRNA, remove_elements

                

    def remove_sample_contaminated(self, dataset : List[Dict], check: bool = False) -> List[Dict]:
        '''Elimina las muestras contaminadas, es decir, la que no contienen el nucleótido A, C, T o G.'''

        clean_final_dataset : List = []
        list_remove: List = []
        count_contaminated = 0

        for record in dataset:
            contaminada: bool = not set(record['seq']).issubset({'A','T','C','G'})
            if not contaminada and record['seq'] != "":
                clean_final_dataset.append(record)
            else:
                list_remove.append(record['old_idx'])
                count_contaminated += 1

        if check:
            assert len(dataset) == len(clean_final_dataset) + count_contaminated
            self._logger.info("Muestras eliminadas en remove_sample_contaminated: ")
            self._logger.info(count_contaminated)
        
        self.dataset = clean_final_dataset
        return clean_final_dataset, list_remove




class Overlap:

    def __init__(self, priority: List[str]):
        self.priority: List[str] = priority

    def handleOverlap(self, gff: DataFrame, check: bool = False):
        # clases que se van a tener en cuenta: genes - elementos transponibles - regiones intergénicas.
        # prioridad lógica: genes > elementos transponibles > regiones intergénicas.
        # Dato: los genes no se van a solapar nunca con las regiones intergénicas. Solo se va a solapar los elementos transponibles con las otras dos clases.

        list_chrs = [grupo.sort_values(by='start').reset_index(drop=True).to_dict(orient="records") for _, grupo in gff.groupby('chr')]

        new_list_records = []
        check_dict = { chromo:{'eliminados': 0, 'fraccionados': 0, 'longitud_menorIgual_cero':0, 'new_record_menorIgual_cero': 0} for chromo in np.unique(gff['chr'])}
        info_data_priority = {'gene-gene': 0, 'gene-transposable_element': 0, 'gene-intergenic_region': 0, 'transposable_element-transposable_element': 0, 'transposable_element-intergenic_region': 0}
        gene_solap_other_genes = []
        info_data_low_priority = {'transposable_element-gene': 0, 'intergenic_region-gene': 0, 'intergenic_region-transposable_element': 0}


        for list_records_chr in list_chrs: # recorremos todos los chr
            i = 0
            j = 1
            stop = len(list_records_chr)

            while j < stop : # recorremos todas las muestras del chr
                actual_record: Dict = list_records_chr[i]
                compare_record: Dict = list_records_chr[j]

                # base case
                if actual_record['end'] < compare_record['start'] or actual_record['type'] == compare_record['type']: # no solapa con nada
                    new_list_records.append(actual_record)
                    i += 1
                    j = i + 1
                # other case (overlap)
                else:
                    # actual record have the priority.
                    if self.priority.index(actual_record['type']) < self.priority.index(compare_record['type']):
                        info_data_priority[actual_record['type']+'-'+compare_record['type']] += 1
                        if actual_record['type'] == 'gene' and compare_record['type'] == 'gene':
                            gene_solap_other_genes.append((actual_record['ID'], compare_record['ID']))
                        if actual_record['end'] >= compare_record['end']:
                            list_records_chr.pop(j)
                            check_dict[actual_record['chr']]['eliminados'] += 1
                        else:
                            compare_record['start'] = actual_record['end'] + 1
                            length_compare_record = int(compare_record['end']) - int(compare_record['start'])
                            if length_compare_record < 0:
                                list_records_chr.pop(j)
                                check_dict[actual_record['chr']]['longitud_menorIgual_cero'] += 1
                                

                    # actual record don't have the priority.
                    elif self.priority.index(actual_record['type']) > self.priority.index(compare_record['type']):
                        info_data_low_priority[actual_record['type']+'-'+compare_record['type']] += 1
                        if actual_record['end'] >= compare_record['end']:
                            new_record = actual_record.copy()
                            actual_record['end'] = compare_record['start'] - 1
                            new_record['start'] = compare_record['end'] + 1
                            length_new_record = int(new_record['end']) - int(new_record['start'])
                            if length_new_record < 0:
                                check_dict[actual_record['chr']]['new_record_menorIgual_cero'] += 1
                            else:
                                list_records_chr.insert(j+1, new_record)
                                check_dict[actual_record['chr']]['fraccionados'] += 1
                            length_actual_record = int(actual_record['end']) - int(actual_record['start'])
                            if length_actual_record < 0:
                                list_records_chr.pop(i)
                                check_dict[actual_record['chr']]['longitud_menorIgual_cero'] += 1
                        else:
                            actual_record['end'] = compare_record['start'] - 1
                            length_actual_record = int(actual_record['end']) - int(actual_record['start'])
                            if length_actual_record < 0:
                                list_records_chr.pop(i)
                                check_dict[actual_record['chr']]['longitud_menorIgual_cero'] += 1
                list_records_chr = sorted(list_records_chr, key=lambda r: int(r['start']))
                j = i + 1
                stop = len(list_records_chr)


            for element in list_records_chr[i:]:
                if element not in new_list_records:
                    new_list_records.append(element)
        
        if check:
            eliminados: int = 0
            fraccionados: int = 0
            longitud_menorIgual_cero: int = 0
            for chr in check_dict.keys():
                eliminados += check_dict[chr]['eliminados']
                fraccionados += check_dict[chr]['fraccionados']
                longitud_menorIgual_cero += check_dict[chr]['longitud_menorIgual_cero']
            assert gff.shape[0] == (((len(new_list_records)- fraccionados) + eliminados) + longitud_menorIgual_cero)
        
        return new_list_records, info_data_priority, info_data_low_priority, gene_solap_other_genes
    


    def overlap_two_records(self, list_actuals_records: List[Dict], other_record: Dict, info_solap: Dict):
        new_actual_records = []
        for actual_record in list_actuals_records:
            middle: bool = False
            # Not overlap.
            if other_record['end'] < actual_record['start'] or other_record['start'] > actual_record['end']:
                new_actual_records.append(actual_record)
                continue
            else:
                # Overlap in the entire sequence.
                if other_record['start'] <= actual_record['start'] and other_record['end'] >= actual_record['end']:
                    info_solap[other_record['type']+'-'+actual_record['type']] += 1
                    continue
                # Overlap in the right of the sequence.
                elif actual_record['start'] <= other_record['start'] <= actual_record['end'] and other_record['end'] >= actual_record['end']:
                    actual_record['end'] = other_record['start'] - 1
                # Overlap in the left of the sequence.
                elif actual_record['start'] <= other_record['end'] <= actual_record['end'] and other_record['start'] <= actual_record['start']:
                    actual_record['start'] = other_record['end'] + 1
                # Overlap in the middle of the sequence.
                else:
                    if actual_record['type'] == 'transposable_element':
                        info_solap['transposable_element'] += 1
                    middle = True

                    left = actual_record.copy()
                    right = actual_record.copy()

                    left['end'] = other_record['start'] - 1
                    right['start'] = other_record['end'] + 1

            # Check if length is negative, otherwise add to list
            if middle:
                if left['start'] <= left['end']:
                    new_actual_records.append(left)
                if right['start'] <= right['end']:
                    new_actual_records.append(right)
            else:
                if actual_record['start'] <= actual_record['end']:
                    new_actual_records.append(actual_record)

        return new_actual_records


                
    
    def handleOverlap_w_priority(self, gff: DataFrame, check: bool = False):

        list_chrs = [grupo.sort_values(by='start').reset_index(drop=True).to_dict(orient="records") for _, grupo in gff.groupby('chr')]
        info_solap = {'gene-transposable_element': 0, 'transposable_element-intergenic_region': 0, 'transposable_element': 0}

        end_records = []

        for list_records_chr in list_chrs:

            list_pending: List[Dict] = []
            list_active: List[Dict] = []
            i: int = 0
            stop: int = len(list_records_chr)

            while i < stop:

                # 1. Processing...
                actual_record = list_records_chr[i]
                forward_overlap = [forward_record for forward_record in list_records_chr[i+1:] if actual_record['end'] >= forward_record['start']]
                list_active = [backward_record for backward_record in list_active if backward_record['end'] >= actual_record['start']] # Se pueden descartar para siempre.
                new_active_records = []
                list_pending_obj_record = [] 
                idx_remove_list_pending = []
                for idx, pending_record in enumerate(list_pending):
                    if pending_record['end'] < actual_record['start']:
                        idx_remove_list_pending.append(idx)
                    elif (actual_record['start'] > pending_record['start']) and (actual_record['end'] >= pending_record['end']) and (pending_record['start'] < actual_record['start'] <= pending_record['end'] ):
                        new_active_records.append(pending_record)
                        idx_remove_list_pending.append(idx)
                    elif not actual_record['end'] < pending_record['start']: # (actual_record['end'] >= pending_record['start']) or (actual_record['start'] <= pending_record['start'] and actual_record['end'] >= pending_record['end']) or (pending_record['start'] <= actual_record['start'] and pending_record['end'] >= actual_record['end'])
                        list_pending_obj_record.append(pending_record)
                list_active.extend(new_active_records)
                for idx in sorted(idx_remove_list_pending, reverse=True):
                    list_pending.pop(idx)

                list_actuals_records = [actual_record.copy()]

                # 1.2 Cut actual record. Only if the priority is higher than actual record.
                for forward_overlap_record in forward_overlap+list_pending_obj_record+list_active:
                    if self.priority.index(forward_overlap_record['type']) < self.priority.index(actual_record['type']): 
                        list_actuals_records = self.overlap_two_records(list_actuals_records, forward_overlap_record, info_solap)

                # 2. Update list_pending, list_active, list_records_chr and end_records.
                for new_record in list_actuals_records:
                    list_pending.append(new_record)
                i += 1
                end_records.extend(list_actuals_records)
        return end_records, info_solap



class Intermediate_data:
    
    def generate_intermediateData(self, dataset: DataFrame, max_len_seq: int, mapping: Dict[str, List], mapping2: Dict[str, List], min_intergenic_data: int = 1000, min_length_seq: int = 3000):
        gen_dataset = [grupo[grupo.type == 'gene'].sort_values(by='start').reset_index(drop=True).to_dict(orient="records") for _, grupo in dataset.groupby('chr')]
        valid_data = []

        for chr in gen_dataset:
            # El último no se introduce pero no importa porque puede llegar a ser mayor que el tamaño del fasta.
            i = 0
            j = 1
            value_end_backward: int = 0
            length_chr = len(chr)
            while j < length_chr:
                actual_data = chr[i]
                compare_data = chr[j]

                # base case
                if (value_end_backward+min_intergenic_data < actual_data["start"]) and (actual_data["end"] < compare_data["start"] - min_intergenic_data) and ((actual_data['end'] - actual_data['start']) > min_length_seq):
                    modify_actual_data = actual_data.copy()
                    modify_actual_data['inferior_limit'] = value_end_backward
                    modify_actual_data['superior_limit'] = compare_data["start"]
                    valid_data.append(modify_actual_data)
                
                value_end_backward = actual_data["end"] if actual_data["end"] > value_end_backward else value_end_backward
                i += 1
                j += 1
                    
        new_data = []
        for record in valid_data:
            inferior_limit = record["inferior_limit"]
            superior_limit = record["superior_limit"]
            del record["inferior_limit"]
            del record["superior_limit"]
            start_record = record["start"]
            end_record = record["end"]

            # 1. Left
            m_minus_left_limit = (start_record+min_length_seq)-max_len_seq if (start_record+min_length_seq)-max_len_seq > inferior_limit else inferior_limit
            x_minus_left_limit = start_record-min_intergenic_data

            
            start_left = random.randint(m_minus_left_limit, x_minus_left_limit)

            m_max_left_limit = start_record+min_length_seq
            x_max_left_limit = end_record if start_left+max_len_seq > end_record else start_left+max_len_seq

            end_left = random.randint(m_max_left_limit, x_max_left_limit)

            record_left = record.copy()
            record_left["start"] = start_left
            record_left["end"] = end_left
            proportion_gen = (end_left - start_record) / (end_left - start_left)
            proportion_ir = 1 - proportion_gen
            record_left['proportions'] = [proportion_ir,proportion_gen]
            record_left['completeness'] = (record_left["end"] - start_record) / (end_record - start_record)

            record_left["type"] = "gene-ir"
            new_data.append(record_left)

            # 2. Right
            x_max_right_limit = superior_limit if (end_record-min_length_seq)+max_len_seq > superior_limit else (end_record-min_length_seq)+max_len_seq
            m_max_right_limit = end_record+min_intergenic_data

            end_right = random.randint(m_max_right_limit, x_max_right_limit)

            m_minus_right_limit = start_record if (end_right-max_len_seq) < start_record else (end_right-max_len_seq)
            x_minus_right_limit = end_record-min_length_seq 

            start_right = random.randint(m_minus_right_limit, x_minus_right_limit)

            record_right = record.copy()
            record_right["start"] = start_right
            record_right["end"] = end_right
            proportion_gen = (end_record - start_right) / (end_right - start_right)
            proportion_ir = 1 - proportion_gen
            record_right['proportions'] = [proportion_ir,proportion_gen]
            record_right['completeness'] = (end_record - record_right['start']) / (end_record - start_record)
            record_right["type"] = "gene-ir"
            new_data.append(record_right)

            # 3. Middle
            if (end_record - start_record) + (2*min_intergenic_data) <= max_len_seq:

                proportions_sides = (max_len_seq - (end_record - start_record)) // 2

                m_minus_middle_limit = inferior_limit if (start_record - proportions_sides) < inferior_limit else (start_record - proportions_sides)
                x_minus_middle_limit = start_record - min_intergenic_data

                start_middle = random.randint(m_minus_middle_limit, x_minus_middle_limit)

                x_right_middle_limit = superior_limit if (end_record + proportions_sides) > superior_limit else (end_record + proportions_sides)
                m_right_middle_limit =  end_record + min_intergenic_data

                end_middle = random.randint(m_right_middle_limit, x_right_middle_limit)

                record_middle = record.copy()
                record_middle["start"] = start_middle
                record_middle["end"] = end_middle
                proportion_gen = (end_record - start_record) / (end_middle - start_middle)
                proportion_ir = 1 - proportion_gen
                record_middle['proportions'] = [proportion_ir,proportion_gen]
                record_middle['completeness'] = (end_record - start_record) / (end_record - start_record)
                record_middle["type"] = "gene-ir"
                new_data.append(record_middle)

                
                

        dataset['proportions'] = dataset['type'].map(mapping)
        dataset['completeness'] = dataset['type'].map(mapping2)

        return pd.concat([pd.DataFrame(new_data), dataset], axis=0)

        
class Gene_limits:

    def generate_intermediate_gene(self, dataset: DataFrame):
        # bins y aleatoriedad entre los valores del bins.
        assert (dataset['type'] == 'gene').all()

        LIMIT: int = 50
        list_record_gene = dataset.to_dict(orient = "records")
        new_records: List[Dict] = []
        for record in list_record_gene:

            length_gene = record['end'] - record['start'] + 1
            percentage_new_sequences = [random.randint(0, 10), random.randint(10, 20), random.randint(20, 30), random.randint(30, 40), random.randint(40, 50), random.randint(50, 60), random.randint(60, 70), random.randint(70, 80), random.randint(80, 90), random.randint(90, 100)]
            length_new_sequences = [ int((percentage*length_gene)/100) for percentage in percentage_new_sequences]
            for  percentage, new_length in zip(percentage_new_sequences, length_new_sequences):
                copy_record = record.copy()
                if percentage > LIMIT:
                    # TODO: cambiar esto por favor.
                    # copy_record['start'] = random.randint(copy_record['start'], copy_record['end'] - new_length + 1)
                    # copy_record['end'] = copy_record['start'] + new_length - 1
                    copy_record['start'] = copy_record['end'] - new_length + 1
                else:
                    value_1 = random.randint(copy_record['start'], copy_record['end'])
                    if value_1+new_length-1 > copy_record['end']:
                        copy_record['end'] = value_1
                        copy_record['start'] = value_1-new_length + 1
                    else:
                        copy_record['start'] = value_1
                        copy_record['end'] = value_1+new_length-1
                copy_record['COMPLETENESS'] = percentage # round((copy_record['end']- copy_record['start'] + 1)/ length_gene, 2)
                new_records.append(copy_record)

        return pd.DataFrame(new_records)