import pandas as pd
import os
import numpy as np
import logging
from Bio import SeqIO
import pyranges as pr

from typing import Dict, List
from pandas import DataFrame


class CleanData:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.dataset: List[Dict] = []


    def obtain_dicc_bed(self, route: str, encoding: str = 'utf-8') -> Dict[int, DataFrame]:
        '''Devuelve todos los cromosomas de la especie junto a su fichero GFF3 como un dataframe'''
        all_bed : Dict[str,DataFrame] = {}
        data = pd.read_csv(route, comment='#', sep='\t', header=None, encoding= encoding)
        data.columns = ['chr','db','type','start','end','score','strand','phase','attributes']
        all_bed[1] = data
        return all_bed
    
    def obtain_dicc_fasta(self, route: str, mapping = None) -> Dict[int, str]:
        '''Devuelve todos los cromosomas de la especie junto a su fichero fasta como un string'''
        all_fasta : Dict[str,str] = {}
        with open(route, 'r') as file:
            for record in SeqIO.parse(file, "fasta"):
                if mapping:
                    all_fasta[mapping[record.id]] = str(record.seq).upper()
                else:
                    all_fasta[record.id] = str(record.seq).upper()
        return all_fasta
    

    def types_type(self, dicc_bed: Dict[int, DataFrame]):
        '''Tipos únicos presentes en los archivos GFF3 de la especie'''
        lista_definitiva : List[str] = []
        for key in dicc_bed.keys():
            bed_df : DataFrame = dicc_bed[key]
            list_types : List[str] = bed_df['type']
            lista_definitiva.extend(list_types)
        self._logger.info(np.unique(lista_definitiva))


    def add_transposable_element(self, gff_complete: Dict[int, DataFrame], route_gff_te: str):
        gff_complete[1].columns = ['Chromosome','db','type','Start','End','score','strand','phase','attributes']
        gr_ann = pr.PyRanges(gff_complete[1])
        data_te = pd.read_csv(route_gff_te, comment='#', sep='\t', header=None, encoding= 'latin-1')
        data_te.columns = ['Chromosome','db','type','Start','End','score','strand','phase','attributes']
        data_te['type'] = "transposable_element"
        gr_TEs = pr.PyRanges(data_te)

        # Restar los TEs de las anotaciones (PyRanges trocea automáticamente los intervalos)
        gr_ann_minus_TE = gr_ann.subtract(gr_TEs)
        print("Antes: ", gff_complete[1].shape)
        print("Después: ", (gr_ann_minus_TE.df).shape)

        # Combinar las anotaciones troceadas con los TEs y ordenar por coordenadas
        gr_merged = pr.concat([gr_ann_minus_TE, gr_TEs]).sort()
        dataset = gr_merged.df
        print("Añado: ", (gr_TEs.df).shape)
        print("Final: ", dataset.shape)
        dataset.columns = ['chr','db','type','start','end','score','strand','phase','attributes']
        gff_complete[1] = dataset
        return gff_complete


    def select_elements_gff(self, selected : List[str], dicc_bed: Dict[int, DataFrame]) -> Dict[int, DataFrame]:    
        '''Selecciona del los dataframes de los archivos GFF3 las clases deseadas.'''
        for key in dicc_bed.keys():
            bed_df : DataFrame = dicc_bed[key]
            clean_bed : DataFrame = bed_df[bed_df.type.isin(selected)]
            dicc_bed[key] = clean_bed
        return dicc_bed


    def extract_cds(self, gff: Dict[int, DataFrame], fasta: Dict[int, str]) -> List[Dict]:
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

        gff[1]['Parent'] = gff[1]['attributes'].str.extract(r'Parent=([^;]+)', expand=False)
        # eliminamos las líneas sin Parent
        gff = gff[1].dropna(subset=['Parent'])
        bed_file = gff.sort_values(['Parent','start'], ascending=[True, True])

        dataset_dict : Dict[str, Dict] = {}
        list_bed : List = bed_file.to_dict(orient='records')
        for record in list_bed:
            fasta_a_usar : str = str(record['chr'])
            if fasta_a_usar not in list(fasta.keys()):
                problemas.append(fasta_a_usar)
                continue
            
            fasta_file : str = fasta[fasta_a_usar]
                
            if dataset_dict.get(fasta_a_usar, -1) == -1:
                dataset_dict[fasta_a_usar] = {}

            if record['start'] > record['end']:
                # logger.info("Paso completamente de esta mierda")
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.'):
                    dataset_dict[fasta_a_usar][record["Parent"]] = dataset_dict[fasta_a_usar].get(record["Parent"], "") + fasta_file[record['start']-1:record['end']]
            elif record['strand'] == '-':
                    dataset_dict[fasta_a_usar][record["Parent"]] = complement(fasta_file[record['start']-1:record['end']]) + dataset_dict[fasta_a_usar].get(record["Parent"], "")

        final_dataset: List[Dict] = [
            {"type": "CDS", "seq":seq}
            for inner_dict in dataset_dict.values()
            for seq       in inner_dict.values()
        ]

        self._logger.info("Chromosomas/scaffolds no encontrados en el archivo fasta: ")
        self._logger.info(np.unique(problemas))
        self.dataset = final_dataset
        return final_dataset

    def clean_cds(self, list_records: List[Dict]):
        new_list_record: List[Dict] = []
        contador = 0
        for record in list_records:
            if 'ATG' != record['seq'][:3] or record['seq'][-3:] not in ("TAA", "TAG", "TGA") or len(record['seq']) % 3 != 0:
                contador += 1
            else:
                new_list_record.append(record)

        self._logger.info(contador)
        return new_list_record

    def extract_sequences_mRNA(self, gff: Dict[int, DataFrame], fasta: Dict[int, str]) -> List[Dict]:
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
        for key in gff.keys():
            gff[key]['Parent'] = gff[key]['attributes'].str.extract(r'Parent=([^;]+)', expand=False)
            gff[key] = gff[key].dropna(subset=['Parent'])
            gff[key]['ID'] = gff[key]['attributes'].str.extract(r'ID=([^;]+)', expand=False)
            gff = gff[key].dropna(subset=['ID'])
            bed_file = gff.sort_values(['Parent','start'], ascending=[True, True])
            list_bed : List = bed_file.to_dict(orient='records')
            parent_type: Dict[str, str] = {}
            for record in list_bed:
                if record['type'] == "mRNA":
                    parent_type[record['Parent']] = "gene"
                parent_type[record['ID']] = record['type']
            for record in list_bed:
                if record['type'] == 'mRNA':
                    continue
                if record['type'] == 'intron' and parent_type.get(record['Parent'], -1) == -1:
                    continue
                elif parent_type.get(record['Parent'], -1) == -1 or parent_type[record['Parent']] != "mRNA":
                    continue
                fasta_a_usar : str = str(record['chr'])
                if fasta_a_usar not in list(fasta.keys()):
                    problemas.append(fasta_a_usar)
                    continue
                fasta_file : str = fasta[fasta_a_usar]
                longitud = (int(record['end']) - int(record['start'])-1)
                if (longitud < minimo) or (longitud > maximo):
                    continue
                if record['start'] > record['end']:
                    # logger.info("Paso completamente de esta mierda")
                    continue
                elif (record['strand'] == '+') or  (record['strand'] == '.'):
                    final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type']})
                elif record['strand'] == '-':
                    final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type']})

        self._logger.info("Chromosomas/scaffolds no encontrados en el archivo fasta: ")
        self._logger.info(np.unique(problemas))
        self.dataset = final_dataset
        return final_dataset
    


    def extract_sequences_counting_chr(self, gff: Dict[int, DataFrame], fasta: Dict[int, str]) -> List[Dict]:
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
        for key in gff.keys():
            bed_file : DataFrame = gff[key] # Solo hay uno.
            list_bed : List = bed_file.to_dict(orient='records')
            for record in list_bed:
                fasta_a_usar : str = str(record['chr'])
                if fasta_a_usar not in list(fasta.keys()):
                    problemas.append(fasta_a_usar)
                    continue
                fasta_file : str = fasta[fasta_a_usar]
                longitud = (int(record['end']) - int(record['start'])-1)
                if (longitud < minimo) or (longitud > maximo):
                    continue
                if record['start'] > record['end']:
                    # logger.info("Paso completamente de esta mierda")
                    continue
                elif (record['strand'] == '+') or  (record['strand'] == '.'):
                    final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type']})
                elif record['strand'] == '-':
                    final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type']})

        self._logger.info("Chromosomas/scaffolds no encontrados en el archivo fasta: ")
        self._logger.info(np.unique(problemas))
        self.dataset = final_dataset
        return final_dataset
    
    def sample_contaminated(self, dataset: List[Dict], types : Dict[str, int]):
        '''Obtiene las muestras que son contaminas, es decir, no contienen el nucleótido A, C, T o G.'''

        conteo : int = 0
        for record in dataset:
            contaminada : bool = not set(record['seq']).issubset({"A", "T", "C", "G"})
            if contaminada:
                conteo = conteo + 1
                types[record['type']] = types[record['type']] + 1
                
        self._logger.info("Muestras contaminadas: "+str(conteo))
        self._logger.info("Porcentaje: "+str((conteo*100)/len(dataset)))

    def remove_sample_contaminated(self, dataset : List[Dict]) -> List[Dict]:
        '''Elimina las muestras contaminadas, es decir, la que no contienen el nucleótido A, C, T o G.'''

        clean_final_dataset : List = []

        for record in dataset:
            contaminada: bool = not set(record['seq']).issubset({'A','T','C','G'})
            if not contaminada:
                clean_final_dataset.append(record)
        
        self._logger.info("Nuevo tamaño del dataset tras limpieza: %d",len(clean_final_dataset))
        self.dataset = clean_final_dataset
        return clean_final_dataset
    
