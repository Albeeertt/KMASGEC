#!/usr/bin/env python

# Typing
from typing import Dict, List

# work open 
import argparse
import os

def obtener_argumentos_pre():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="", help="GPUs a usar, e.g. '0', '0,1', '0,2,3'", required=True)
    known, _ = parser.parse_known_args()
    return known

pre_args = obtener_argumentos_pre()

os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.gpus
from functools import partial
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pkg_resources
import pandas as pd
import json
import gc
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# work close
from kmasgec.core.CleanData import CleanData, Modify_samples
from kmasgec.core.GenerateDataset import GenerateDataset
from kmasgec.utils.agat import Agat
from kmasgec.utils.json_pytorch import save_all_to_json
from kmasgec.core.models.loaders.Loader import Base64JSONIterableDataset, collate_fn_oneHead
from kmasgec.core.models.epochs.epoch import iteration_test_oneHead
from kmasgec.core.models.model_architecture.transformers import TransformerClassifier_attnPool, TransformerClassifier_attnPool_CrossAttn
from kmasgec.utils.plots.sections.section_gen import Gen
from kmasgec.utils.plots.sections.section_ir import IntergenicRegion
from kmasgec.utils.plots.sections.section_summary import Summary
from kmasgec.utils.CreateGFF import CreateGFF

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    parser.add_argument('--batch_size', type=int, required=True, help = "Tamaño del batch size")
    parser.add_argument('--out', type=str, required=True, help="")
    parser.add_argument("--model", type=str, required=True, help="Modelo a usar.")
    parser.add_argument('--add_labels', action='store_true', help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--fine_tunning', action='store_true', help="")
    parser.add_argument('--train', action='store_true', help="Si deseas entrenar un modelo desde cero")
    parser.add_argument('--gpus', type=str, default="", help="GPUs a usar, e.g. '0', '0,1', '0,2,3'", required=True) # TODO: ignorar, hacer un único parser y ya.
    parser.add_argument("--lens_mode", action="store_true", help="Divide las secuencias en trozos.")
    parser.add_argument("--max_len_seq", type=int, required=False, help="tamaño máximo de la secuencia.")

    # Analizar los argumentos pasados por el usuario
    return parser.parse_args()


def ejecutar():
    time_inicio_crearDataset = time.time()
    NAME_HTML: str = 'info.html'

    agrupacion = 3
    kmer: bool = True

    args = obtener_argumentos()

    if args.max_len_seq:
        MAX_LEN_SEQ = args.max_len_seq
    else:
        MAX_LEN_SEQ = 10_000 


    route_out = args.out
    if not os.path.exists(route_out):
        os.mkdir(route_out)
        
    route_out = route_out+'/' if not route_out.endswith('/') else route_out
        
    if args.add_labels:
        instance_agat = Agat("katulu")
        new_route_gff = instance_agat.add_introns(args.gff, route_out)
        new_route_gff = instance_agat.add_intergenicRegion(new_route_gff, route_out)
        args.gff = instance_agat.keep_longest_isoform(new_route_gff, route_out)

    ruta_data_first_algorithm = route_out+'first.json'
    ruta_data_gff = args.gff
    ruta_data_fasta = args.fasta

    instance_cleanData = CleanData()
    instance_modify_samples = Modify_samples()
    gff = instance_cleanData.obtain_gff(ruta_data_gff, encoding='latin-1')
    elements_plus_te_mRNA, remove_idx_mRNA = instance_cleanData.obtain_gene_w_mRNA(gff, ['intergenic_region'], False, False)
    dataframe_elements_plus_te_mRNA = pd.DataFrame(elements_plus_te_mRNA)
    dataframe_elements_plus_te_mRNA = instance_modify_samples.change_strand(dataframe_elements_plus_te_mRNA, type_record = 'intergenic_region', new_strand = '-')
    if args.lens_mode:
        dataframe_elements_plus_te_mRNA = instance_modify_samples.lens_mode(dataframe_elements_plus_te_mRNA, MAX_LEN_SEQ)
    fasta = instance_cleanData.obtain_dicc_fasta(ruta_data_fasta)

    data_first_algorithm = dataframe_elements_plus_te_mRNA[dataframe_elements_plus_te_mRNA['type'].isin(['intergenic_region', 'gene'])].copy()
    data_first_algorithm[['start','end']] = data_first_algorithm[['start','end']].apply(pd.to_numeric, errors='coerce')

    remove_idx_chr = []
    remove_idx_startEnd = []
    remove_contaminated = []
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    data_first_algorithm['new_idx'] = data_first_algorithm.index

    for sample in data_first_algorithm.to_dict(orient='records'):
        new_record, add_idx_chr, add_idx_startEnd = instance_cleanData.extract_sample_counting_chr(sample, fasta)
        if new_record is None:
            remove_idx_chr.extend(add_idx_chr)
            remove_idx_startEnd.extend(add_idx_startEnd)
            continue
        if instance_cleanData.is_contaminated(new_record):
            remove_contaminated.append(new_record['old_idx'])
            continue
        new_record['new_idx'] = sample['new_idx']
        X = []
        y = []
        place = []
        place_new = []
        seq = [vocab[nucleotide] for nucleotide in new_record['seq']]
        X.append(seq)
        y.append(np.array(1) if new_record['type'] == 'gene'
            else np.array(0) if new_record['type'] == "intergenic_region"
            else -1)
        place.append(new_record['old_idx'])
        place_new.append(new_record['new_idx'])

        X_fin = [np.asarray(i, dtype=np.float32) for i in X]
        y_fin = [np.asarray(i, dtype=np.float32) for i in y]
        place_fin = [np.asarray(i, dtype=np.int64) for i in place]
        place_new_fin = [np.asarray(i, dtype=np.int64) for i in place_new]

        save_all_to_json(X_fin, y_fin, place_fin, place_new_fin, filename=ruta_data_first_algorithm, names=['X', 'Y', 'Place', 'Place_new'])

    time_fin_crearDataset = time.time()

    # Model 1
    # ---------------------------------------------------------------------------------------------

    time_inicio_algoritmo = time.time()

    batch_size: int = args.batch_size
    min_len_seq: Dict[int, int] = {0: 50, 1: 50}
    instance_generateDataset  = GenerateDataset(False, agrupacion, kmer)
    padding_value = len(instance_generateDataset.vocabularyComplete)
    vocab_size = len(instance_generateDataset.vocabularyComplete)+1
    print("Tamaño del vocabulario: ", len(instance_generateDataset.vocabularyComplete))
    partial_collateFN = partial(collate_fn_oneHead, padding_value=padding_value, max_padding=MAX_LEN_SEQ)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    device_cuda: bool = False
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache() 
        print("Usando GPU de Apple (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        device_cuda = True
        print("Usando NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Usando CPU")

    print("Cargando modelo...")

    model = TransformerClassifier_attnPool( 
        vocab_size=vocab_size,
        padding_idx=padding_value,
        embed_dim=256, 
        num_heads=8,
        num_layers=3, 
        dim_feedforward=1024, 
        num_classes=2, 
        dropout=0.2,
    )
    torch.compile(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(pkg_resources.resource_filename("kmasgec", f"generate_models/{args.model}"), map_location=device)
    state = checkpoint['model_state_dict']
    model.load_state_dict(state, strict=True)

    if len(pre_args.gpus.split(',')) > 1:
        model = nn.DataParallel(model)

    
    dataset = Base64JSONIterableDataset(ruta_data_first_algorithm, min_len_seq, MAX_LEN_SEQ, instance_generateDataset, kmer = kmer)
    loader_test  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=True,
        collate_fn=partial_collateFN
    )

    n_batches_test = len(loader_test)

    pbar_test = tqdm(loader_test, total=n_batches_test, desc="Test")
    report_dict, _, all_preds, all_places, all_places_new, all_softmax_official_values = iteration_test_oneHead(pbar_test,  model, device, criterion)
    pbar_test.close()
    os.remove(ruta_data_first_algorithm)

    model.to('cpu')
    del model
    
    gc.collect()
    if device_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    time_fin_algoritmo = time.time()
    time_inicio_postProcesado = time.time()

    instance_createGFF = CreateGFF(gff, all_preds, all_places, all_softmax_official_values)
    instance_createGFF.create_gff(remove_idx_mRNA, remove_idx_chr, remove_idx_startEnd, remove_contaminated, route_out, three_columns=False)

    # -------------------------------
    # TODO: borrar después
    data_first_algorithm = data_first_algorithm.set_index('new_idx')
    data_first_algorithm['Result'] = 'None'
    data_first_algorithm['prob_gene'] = np.nan
    data_first_algorithm['prob_intergenic_region'] = np.nan
    probs = torch.softmax(torch.tensor(all_softmax_official_values), dim=1)
    new_prob_ir = [element[0] for element in probs]
    new_prob_gene = [element[1] for element in probs]
    prob_ir = np.asarray(new_prob_ir, np.float16)
    prob_gene = np.asarray(new_prob_gene, np.float16)
    preds_names = np.vectorize({
            0: 'intergenic_region',
            1: 'gene'
        }.get)(all_preds)
    a_places = np.asarray(all_places_new, np.int64)
    data_first_algorithm.loc[a_places, 'Result'] = preds_names
    data_first_algorithm.loc[a_places, 'prob_gene'] = prob_gene
    data_first_algorithm.loc[a_places, 'prob_intergenic_region'] = prob_ir
    data_first_algorithm.to_csv(route_out+"PREV_result.csv", sep=',')
    # -------------------------------

    with open(route_out+'report.json', "a") as f:
        json.dump(report_dict, f, indent=4)

    time_fin_postProcesado = time.time()

    dict_times = {
        'tiempo_CrearDataset': (time_fin_crearDataset-time_inicio_crearDataset)/60,
        'tiempo_algoritmo': (time_fin_algoritmo-time_inicio_algoritmo)/60,
        'tiempo_postProcesado': (time_fin_postProcesado-time_inicio_postProcesado)/60
                  }

    with open(route_out+'time.json', 'a') as f:
        json.dump(dict_times, f, indent=4)