#!/usr/bin/env python

# Typing
from typing import Dict, List

# work open 
import argparse
import os
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
from kmasgec.utils.json_pytorch import save_chunks_andIdx_to_json
from kmasgec.models.loaders.Loader import Base64JSONIterableDataset, collate_fn_oneHead
from kmasgec.models.epochs.epoch import iteration_test_oneHead_w_reject, iteration_train_oneHead
from kmasgec.models.model_architecture.transformers import TransformerClassifier

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    parser.add_argument('--add_labels', action='store_true', help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--out', type=str, required=True, help="")
    parser.add_argument('--fine_tunning', action='store_true', help="")
    parser.add_argument('--train', action='store_true', help="Si deseas entrenar un modelo desde cero")
    parser.add_argument('--gpu', action='store_true', help="")

    
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

    seleccionados: List[str] = ['CDS']
    seleccionados2: List[str] = ['five_prime_UTR', 'three_prime_UTR', 'intron', 'mRNA']
    seleccionados3: List[str] = ['intergenic_region', 'transposable_element']

    file_specie_test: str = route_out+'/test.json'
    #TODO: borrar fichero creado.
    ruta_data_bed = args.gff
    ruta_data_fasta = args.fasta

    check: bool = True

    instance_cleanData = CleanData()
    gff = instance_cleanData.obtain_gff(ruta_data_bed, encoding='latin-1')
    # gff[1].to_csv(route_out+'result.csv', index=False)
    gff_copy = instance_cleanData.obtain_gff(ruta_data_bed, encoding='latin-1')
    gff_copy2 = instance_cleanData.obtain_gff(ruta_data_bed, encoding='latin-1')
    fasta = instance_cleanData.obtain_dicc_fasta(ruta_data_fasta)
    gff = instance_cleanData.select_elements_gff(seleccionados, gff, check)
    list_records = instance_cleanData.extract_cds(gff, fasta, check)
    print("longitud extraccion: ", len(list_records))
    list_clean_records = instance_cleanData.remove_sample_contaminated(list_records, check)
    print("longitud remove cds: ", len(list_clean_records))
    list_clean_cds = instance_cleanData.clean_cds(list_clean_records, check)
    print("longitud clean cds: ", len(list_clean_cds))
    
    gff_copy = instance_cleanData.select_elements_gff(seleccionados2, gff_copy, check)
    list_records = instance_cleanData.extract_sequences_mRNA(gff_copy, fasta, check)
    list_clean_records_mRNA = instance_cleanData.remove_sample_contaminated(list_records, check)

    gff_copy2 = instance_cleanData.select_elements_gff(seleccionados3, gff_copy2, check)
    list_records = instance_cleanData.extract_sequences_counting_chr(gff_copy2, fasta, check)
    list_clean_records = instance_cleanData.remove_sample_contaminated(list_records, check)
    
    data = []
    data.extend(list_clean_cds)
    data.extend(list_clean_records_mRNA)
    data.extend(list_clean_records)

    del fasta
    del gff
    del list_records
    del list_clean_records
    del list_clean_cds
    del list_clean_records_mRNA

    for record in data:
        if record['type'] == 'five_prime_UTR' or record['type'] == 'three_prime_UTR':
            record['type'] = 'UTR'

    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    X = []
    y = []
    identificadores = []
    for record in data:
        seq = [vocab[nucleotide] for nucleotide in record['seq']]
        X.append(seq)
        y.append(2 if record['type'] == "UTR"
                else 1 if record['type'] == "CDS"
                else 0 if record['type'] == "intron"
                else 4 if record['type'] == "transposable_element"
                else -1 if record['type'] == "intergenic_region"
                else -2) 
        identificadores.append(record['old_idx'])
    
    del data
    X_list = [np.asarray(x, dtype=np.float32) for x in X]  # lista de arrays
    Y_list = [np.asarray(ya, dtype=np.float32) for ya in y]
    identificadores_list = [np.asarray(idx, dtype=np.float32) for idx in identificadores]

    del X; del y; del identificadores

    save_chunks_andIdx_to_json(X_list, Y_list, identificadores_list, file_specie_test)

    # ---------------------------------------------------------------------------------------------

    batch_size = 1
    weights = torch.tensor([1., 1., 1., 1.])
    agrupacion = 3
    instance_generateDataset  = GenerateDataset(False, agrupacion)
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    learning_rate = 1e-5 if args.fine_tunning else 5e-5
    weight_decay=1e-2
    model = TransformerClassifier(
        vocab_size=len(instance_generateDataset.vocabularyComplete)+1,
        padding_idx=len(instance_generateDataset.vocabularyComplete),
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=4,
        max_seq_len=4000,
        dropout=0.3
    ).to(device)
    if args.gpu:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)
    if not args.train and args.gpu:
        checkpoint = torch.load(pkg_resources.resource_filename("kmasgec", "generate_models/model.pt"), map_location=device) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif not args.train and not args.gpu:
        checkpoint = torch.load(pkg_resources.resource_filename("kmasgec", "generate_models/model.pt"), map_location=device)
        model_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            nk = k.replace('module.', '')  # elimina módulo paralelo
            model_state_dict[nk] = v
        optimizer_state_dict = {}
        for k, v in checkpoint["optimizer_state_dict"].items():
            nk = k.replace('module.', '')  # elimina módulo paralelo
            optimizer_state_dict[nk] = v
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    max_len_seq = 12000
    partial_collateFN = partial(collate_fn_oneHead, padding_value=len(instance_generateDataset.vocabularyComplete))

    dataset = Base64JSONIterableDataset(file_specie_test, max_len_seq, instance_generateDataset)
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

    if args.train:
        epochs = 40
        for epoch in range(1, epochs+1):
            pbar_train = tqdm(loader_test, total=n_batches_test, desc="Training")
            iteration_train_oneHead(pbar_train, epoch, model, device, criterion, optimizer)
            pbar_train.close()

    elif args.fine_tunning:
        pbar_test = tqdm(loader_test, total=n_batches_test, desc="Fine Tunning")
        iteration_train_oneHead(pbar_test, 1, model, device, criterion, optimizer)
        pbar_test.close()
    else:
        pbar_test = tqdm(loader_test, total=n_batches_test, desc="Test")
        cm = iteration_test_oneHead_w_reject(pbar_test, model, device, criterion, 4)
        pbar_test.close()


