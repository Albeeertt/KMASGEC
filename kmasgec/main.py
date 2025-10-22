#!/usr/bin/env python

# Typing
from typing import Dict, List

# work open 
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from functools import partial
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pkg_resources
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# work close
from kmasgec.core.CleanData import CleanData
from kmasgec.core.GenerateDataset import GenerateDataset
from kmasgec.utils.agat import Agat
from kmasgec.utils.json_pytorch import save_all_to_json
from kmasgec.core.models.loaders.Loader import Base64JSONIterableDataset, collate_fn_oneHead
from kmasgec.core.models.epochs.epoch import iteration_test_oneHead
from kmasgec.core.models.model_architecture.transformers import TransformerClassifier_pool

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    parser.add_argument('--batch_size', type=int, required=True, help = "Tamaño del batch size")
    parser.add_argument('--out', type=str, required=True, help="")
    parser.add_argument('--add_labels', action='store_true', help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--fine_tunning', action='store_true', help="")
    parser.add_argument('--train', action='store_true', help="Si deseas entrenar un modelo desde cero")
    # parser.add_argument('--gpu', action='store_true', help="")

    
    # Analizar los argumentos pasados por el usuario
    return parser.parse_args()


def ejecutar():
    args = obtener_argumentos()

    if args.add_labels:
        route_out: str = ("/".join(args.gff.split("/")[:-1]))+"/"
        instance_agat = Agat("katulu")
        new_route_gff = instance_agat.add_introns(args.gff, route_out)
        new_route_gff = instance_agat.add_intergenicRegion(new_route_gff, route_out)
        args.gff = instance_agat.keep_longest_isoform(new_route_gff, route_out) #TODO: borrar esto

    route_out = args.out
    ruta_data_first_algorithm = route_out+'first.json'
    ruta_data_second_algorithm = route_out+'second.json'
    ruta_data_gff = args.gff
    ruta_data_fasta = args.fasta

    instance_cleanData = CleanData()
    gff = instance_cleanData.obtain_gff(ruta_data_gff, encoding='latin-1')
    fasta = instance_cleanData.obtain_dicc_fasta(ruta_data_fasta)

    elements_plus_te_mRNA, remove_elements = instance_cleanData.obtain_gene_w_mRNA(gff, ['intergenic_region'], False, False)
    dataframe_elements_plus_te_mRNA = pd.DataFrame(elements_plus_te_mRNA)


    # First Data
    # ---------------------------------------------------------------------------------------------


    data_first_algorithm = dataframe_elements_plus_te_mRNA[dataframe_elements_plus_te_mRNA['type'].isin(['intergenic_region', 'gene'])]

    data_first_algorithm[['start','end']] = data_first_algorithm[['start','end']].apply(pd.to_numeric, errors='coerce')
    data_first_algorithm = (
    data_first_algorithm
      .drop_duplicates(subset=['chr','type','start','end'])
      .loc[lambda df: df['end'] >= df['start']]
      .reset_index(drop=True)
    )

    data_first_algorithm['proportions'] = 1
    data_first_algorithm['COMPLETENESS'] = 1


    list_records, remove_samples_chr, remove_samples_startEnd = instance_cleanData.extract_sequences_counting_chr(data_first_algorithm, fasta)
    list_clean_records : List[Dict] = instance_cleanData.remove_sample_contaminated(list_records)

    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    X = []
    y = []
    proportions = []
    place = []
    for record in list_clean_records:
        seq = [vocab[nucleotide] for nucleotide in record['seq']]
        X.append(seq)
        y.append(1 if record['type'] == "gene"
            else 0 if record['type'] == "intergenic_region"
            else -1) # región intergénica / elemento transponible
        place.append(record['old_idx'])

    X_fin = [np.asarray(i, dtype=np.float32) for i in X]
    y_fin = [np.asarray(i, dtype=np.float32) for i in y]
    place_fin = [np.asarray(i, dtype=np.int32) for i in place]
    save_all_to_json(X_fin, y_fin, place_fin, filename=ruta_data_first_algorithm, names=['X', 'Y', 'Place'])


    # ---------------------------------------------------------------------------------------------

    # Model 1
    # ---------------------------------------------------------------------------------------------

    batch_size: int = args.batch_size
    min_len_seq: Dict[int, int] = {0: 50, 1: 50, 2: 50, 3: 50}
    agrupacion = 6
    kmer: bool = False
    instance_generateDataset  = GenerateDataset(False, agrupacion, kmer)
    vocab_size = len(instance_generateDataset.vocabularyComplete)+1
    padding_value = len(instance_generateDataset.vocabularyComplete)
    print("Tamaño del vocabulario: ", len(instance_generateDataset.vocabularyComplete))
    partial_collateFN = partial(collate_fn_oneHead, padding_value=padding_value)
    proportions = True


    max_len_seq = 100000
    learning_rate = 2e-4
    weight_decay= 5e-3
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    print(device)
    print("Cargando modelo...")
    model =  TransformerClassifier_pool (
        vocab_size=vocab_size,
        padding_idx=padding_value,
        embed_dim=256, # 256
        num_heads=8,
        num_layers=4, # 4
        dim_feedforward=1024, # 3072
        num_classes=2, # Multi class problem (gene, intergenic_region) 
        dropout=0.2,
        pooling = "cls_token"
    )
    model = model.to(device)



    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim == 1 or "bias" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        eps=1e-6,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(pkg_resources.resource_filename("kmasgec", "generate_models/first_obj.pt"), map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    dataset = Base64JSONIterableDataset(ruta_data_first_algorithm, min_len_seq, max_len_seq, instance_generateDataset,kmer = kmer, proportions = proportions)
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
    cm, all_trues, all_preds, all_places = iteration_test_oneHead(pbar_test,  model, device, criterion, 2)
    pbar_test.close()


    print("Next")
    gff['Result'] = '0'
    gff['Bad'] = 'No'
    print("completando")

    for true, pred, old_idx in zip(all_trues, all_preds, all_places):
        gff.loc[old_idx, 'Result'] = pred
        if true != pred:
            gff['Bad'] = 'Yes'
    print("go go go")

    gff.to_csv(route_out+'prueba.csv', sep=',')


        # Second Data
    # ---------------------------------------------------------------------------------------------


    # data_second_algorithm = dataframe_elements_plus_te_mRNA[dataframe_elements_plus_te_mRNA['type'].isin(['intron', 'three_prime_UTR', 'five_prime_UTR', 'CDS'])]

    # data_second_algorithm[['start','end']] = data_second_algorithm[['start','end']].apply(pd.to_numeric, errors='coerce')
    # data_second_algorithm = (
    # data_second_algorithm
    #   .drop_duplicates(subset=['chr','type','start','end'])
    #   .loc[lambda df: df['end'] >= df['start']]
    #   .reset_index(drop=True)
    # )

    # data_second_algorithm['proportions'] = 1
    # data_second_algorithm['COMPLETENESS'] = 1


    # list_records, remove_samples_chr, remove_samples_startEnd = instance_cleanData.extract_sequences_counting_chr(data_second_algorithm, fasta)
    # list_clean_records : List[Dict] = instance_cleanData.remove_sample_contaminated(list_records)

    # vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # X = []
    # y = []
    # place = []
    # for record in list_clean_records:
    #     seq = [vocab[nucleotide] for nucleotide in record['seq']]
    #     X.append(seq)
    #     y.append(1 if record['type'] == "intron"
    #         else 0 if record['type'] == "CDS"
    #         else 2 if record['type'] == "three_prime_UTR"
    #         else 3 if record['type'] == "five_prime_UTR"
    #         else -1) # región intergénica / elemento transponible
    #     place.append(record['old_idx'])
        

    # X_fin = [np.asarray(i, dtype=np.float32) for i in X]
    # y_fin = [np.asarray(i, dtype=np.float32) for i in y]
    # place_fin = [np.asarray(i, dtype=np.int32) for i in place]
    # save_all_to_json(X_fin, y_fin, place_fin, filename=ruta_data_second_algorithm, names=['X', 'Y', 'Place'])


    # # Model 2
    # # ---------------------------------------------------------------------------------------------

    # batch_size: int = args.batch_size
    # agrupacion = 3
    # kmer: bool = False
    # instance_generateDataset  = GenerateDataset(False, agrupacion, kmer)
    # vocab_size = len(instance_generateDataset.vocabularyComplete)+1
    # padding_value = len(instance_generateDataset.vocabularyComplete)
    # print("Tamaño del vocabulario: ", len(instance_generateDataset.vocabularyComplete))
    # partial_collateFN = partial(collate_fn_oneHead, padding_value=padding_value)
    # min_len_seq: Dict[int, int] = {0: 50, 1: 50}


    # max_len_seq = 50000
    # learning_rate = 2e-4
    # weight_decay= 5e-3
    # #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    # torch.cuda.empty_cache()
    # print(device)
    # print("Cargando modelo...")
    # model =  TransformerClassifier_pool (
    #     vocab_size=vocab_size,
    #     padding_idx=padding_value,
    #     embed_dim=256, # 256
    #     num_heads=8,
    #     num_layers=4, # 4
    #     dim_feedforward=1024, # 3072
    #     num_classes=3, # Multi class problem (gene, intergenic_region) 
    #     # max_seq_len=1001,
    #     dropout=0.2,
    #     pooling = "cls_token"
    # )
    # model = model.to(device)


    # decay, no_decay = [], []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if param.ndim == 1 or "bias" in name or "norm" in name:
    #             no_decay.append(param)
    #         else:
    #             decay.append(param)

    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": decay, "weight_decay": weight_decay},
    #         {"params": no_decay, "weight_decay": 0.0},
    #     ],
    #     lr=learning_rate,
    #     eps=1e-6,
    # )

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # criterion = nn.CrossEntropyLoss()

    # checkpoint = torch.load(pkg_resources.resource_filename("kmasgec", "generate_models/second_obj.pt"), map_location=device) 
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    # dataset = Base64JSONIterableDataset(ruta_data_second_algorithm, min_len_seq, max_len_seq, instance_generateDataset,kmer = kmer, proportions = proportions)

    # loader_test  = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=1,
    #     prefetch_factor=1,
    #     persistent_workers=True,
    #     collate_fn=partial_collateFN
    # )

    # n_batches_test = len(loader_test)


    # pbar_test = tqdm(loader_test, total=n_batches_test, desc="Test")
    # cm, all_trues, all_preds, all_places = iteration_test_oneHead(pbar_test,  model, device, criterion, 3)
    # pbar_test.close()

