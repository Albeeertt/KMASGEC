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
        return final_dataset, problemas_chr, remove_samples
    
    def extract_sample_counting_chr(self, record: Dict, fasta: Dict[str, str]) -> List[Dict]:
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
        new_record : Dict = None
        remove_samples = []

        fasta_a_usar : str = str(record['chr'])
        if fasta_a_usar not in list(fasta.keys()):
            problemas_chr.append(record['old_idx'])
        fasta_file : str = fasta[fasta_a_usar]
        if record['start'] > record['end']:
            remove_samples.append(record['old_idx'])
        elif (record['strand'] == '+') or  (record['strand'] == '.'):
            new_record = {'seq': fasta_file[record['start']-1:record['end']], 'type': record['type'], 'old_idx': record['old_idx']}
        elif record['strand'] == '-':
            new_record = {'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type'], 'old_idx': record['old_idx']}
        
        return new_record, problemas_chr, remove_samples

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
            # Tienen que tener padre, a si que se eliminan chr, ri y genes; solo quedan exones, CDS, UTRs, intrones y mRNA o cosas varias. Como se pone la condición de que el padre debe de ser mRNA entonces se busca las subpartes del mRNA.
            if dict_ids_record.get(record['Parent'], -1) != -1 and dict_ids_record[record['Parent']]['type'] == 'mRNA':
                records_genes_produce_mRNA.append(record)
            # Si es mRNA también entra, claro.
            elif record['type'] == "mRNA":
                records_genes_produce_mRNA.append(record)
            # Si es gen y produce mRNA también lo metemos.
            elif record['type'] == "gene" and gene_mRNA_record.get(record['ID'], -1) != -1:
                records_genes_produce_mRNA.append(record)
            # si es una subparte, el padre es un gen y el padre produce mRNA (seguramente un intrón mal anotado), pues para dentro
            elif dict_ids_record.get(record['Parent'], -1) != -1 and dict_ids_record[record['Parent']]['type'] == 'gene' and gene_mRNA_record.get(record['Parent'], -1) != -1 and record['type'] != 'transcript':
                records_genes_produce_mRNA.append(record)
            elif record['type'] != 'intergenic_region':
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
    
    def is_contaminated(self, record: Dict) -> List[Dict]:
        '''Elimina las muestras contaminadas, es decir, la que no contienen el nucleótido A, C, T o G.'''

        contaminada: bool = not set(record['seq']).issubset({'A','T','C','G'})
        if not contaminada and record['seq'] != "":
            return False
        else:
            return True

    def gen_into_ri(self, dataset: DataFrame, limite: int = 1000000, length_ir: int = 15) -> DataFrame:

        list_dataset = dataset.to_dict(orient='records')
        list_dataset = sorted(list_dataset, key= lambda x: x['start'])
        new_list_dataset = []
        i = 0
        contador_si = 0
        contador_no = 0

        while i < len(list_dataset)-2: # para evitar una comprobación dentro casi inutil.
            record = list_dataset[i]
            new_list_dataset.append(record)
            if record['type'] != 'intergenic_region':
                i += 1
                continue
            record_gene = list_dataset[i+1]
            record_ir = list_dataset[i+2]
            n_record = record.copy()
            n_record['type'] = 'ir_into_gen'
            n_record['end'] = record_ir['end']
            if (record_gene['type'] == 'gene') and (record_ir['type'] == 'intergenic_region') and (n_record['end'] - n_record['start'] <= limite) and (record_ir['end'] - record_ir['start'] > length_ir) and (record['end'] - record['start'] > length_ir):    
                new_list_dataset.append(n_record)
                contador_si += 1
            else:
                contador_no += 1
            i += 1
        return pd.DataFrame(new_list_dataset), contador_si, contador_no
    



class Modify_samples:

    # TODO: faltan las etiquetas de cada nucleótido.
    def case_extremes(self, record: Dict, value: str, selected: str, length_new_extreme: int, label_left: str, label_right: str):
        difference: int = 5
        match selected:
            case 'all_gene':
                if value == 'end':
                    record[label_right] = [1 for _ in range(length_new_extreme)]
                elif value == 'start':
                    record[label_left] = [1 for _ in range(length_new_extreme)]
                return record
            case 'all_ir':
                if value == 'end':
                    record[value] = record[value] + length_new_extreme
                    record[label_right] = [0 for _ in range(length_new_extreme)]
                elif value == 'start':
                    record[value] = record[value] - length_new_extreme
                    record[label_left] = [0 for _ in range(length_new_extreme)]
                return record
            case 'half':
                middle_value = int(length_new_extreme / 2)
                finish_value = random.randint(middle_value-difference, middle_value+difference)
                if value == 'end':
                    record[value] = record[value] + finish_value
                    record[label_right] = [1 for _ in range(length_new_extreme-finish_value)]
                    record[label_right].extend([0 for _ in range(finish_value)])
                elif value == 'start':
                    record[value] = record[value] - finish_value
                    record[label_left] = [0 for _ in range(finish_value)]
                    record[label_left].extend([1 for _ in range(length_new_extreme-finish_value)])
                return record
            case 'more_gene':
                more_gene_value = int(0.2 * length_new_extreme)
                finish_value = random.randint(more_gene_value-difference, more_gene_value+difference)
                if value == 'end':
                    record[value] = record[value] + finish_value
                    record[label_right] = [1 for _ in range(length_new_extreme-finish_value)]
                    record[label_right].extend([0 for _ in range(finish_value)])
                elif value == 'start':
                    record[value] = record[value] - finish_value
                    record[label_left] = [0 for _ in range(finish_value)]
                    record[label_left].extend([1 for _ in range(length_new_extreme-finish_value)])
                return record
            case 'more_ir':
                more_ir_value = int(0.8 * length_new_extreme)
                finish_value = random.randint(more_ir_value-difference, more_ir_value+difference)
                if value == 'end':
                    record[value] = record[value] + finish_value
                    record[label_right] = [1 for _ in range(length_new_extreme-finish_value)]
                    record[label_right].extend([0 for _ in range(finish_value)])
                elif value == 'start':
                    record[value] = record[value] - finish_value
                    record[label_left] = [0 for _ in range(finish_value)]
                    record[label_left].extend([1 for _ in range(length_new_extreme-finish_value)])
                return record

    def new_extremes_genes(self, dataset: DataFrame, length_new_extreme: int, label_left: str, label_right: str):

        new_list_dataset = []
        list_dataset = dataset.to_dict(orient='records')
        options = ['all_gene', 'all_ir', 'half', 'more_gene', 'more_ir']
        for idx, record in enumerate(list_dataset):
            if record['type'] == 'gene':
                selected_left = random.choice(options)
                selected_right = random.choice(options)
                print(selected_left)
                print(selected_right)
                print("----------------")
                if idx == 2:
                    break
                record = self.case_extremes(record, 'start', selected_left, length_new_extreme, label_left, label_right)
                record = self.case_extremes(record, 'end', selected_right, length_new_extreme, label_left, label_right)
            new_list_dataset.append(record)
        return pd.DataFrame(new_list_dataset)
    

    def lends_mode(self, dataset: DataFrame, limit: int, zoom: int):

        list_dataset: List[Dict] = dataset.to_dict(orient='records')
        new_list_dataset: List[Dict] = []

        for record in list_dataset:
            if (record['end'] - record['start'] >= limit) and (record['type'] == 'gene' or record['type'] == 'intergenic_region'):
                start_inicial = record['start']
                i: int = 0
                while ((i*zoom)+start_inicial) < record['end']:
                    record_copy = record.copy()
                    record_copy['start'] = (zoom*i)+start_inicial
                    record_copy['end'] = (record_copy['start'] + zoom) if (record_copy['start'] + zoom) <= record['end'] else record['end']
                    new_list_dataset.append(record_copy)
                    i += 1
            else:
                new_list_dataset.append(record)

        new_dataset = pd.DataFrame(new_list_dataset)
        return new_dataset

    def change_strand(self, dataset: DataFrame, type_record: str, new_strand: str = '-'):

        list_dataset: List[Dict] = dataset.to_dict(orient='records')
        new_list_dataset: List[Dict] = []

        for record in list_dataset:
            new_list_dataset.append(record)
            if record['type'] == type_record:
                record_copy = record.copy()
                record_copy['strand'] = new_strand
                new_list_dataset.append(record_copy)

        return pd.DataFrame(new_list_dataset)
