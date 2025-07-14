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
import torch.functional as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# work close
from kmasgec.core.CleanData import CleanData
from kmasgec.core.GenerateDataset import GenerateDataset
from kmasgec.utils.agat import Agat
from kmasgec.utils.json_pytorch import save_chunks_to_json
from kmasgec.models.loaders.Loader import Base64JSONIterableDataset, collate_fn_oneHead
from kmasgec.models.epochs.epoch import iteration_test_oneHead_w_reject, iteration_train_oneHead
from kmasgec.models.model_architecture.transformers import TransformerClassifier

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    parser.add_argument('--add_labels', type=bool, required=False, help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--out', type=str, required=True, help="")
    parser.add_argument('--fine_tunning', type=bool, required=False, help="")
    parser.add_argument('--train', type=bool, required=False, help="Si deseas entrenar un modelo desde cero")

    
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
    try:
        ruta_data_bed = args.gff
        ruta_data_fasta = args.fasta

        instance_cleanData = CleanData()
        gff = instance_cleanData.obtain_dicc_bed(ruta_data_bed, encoding='latin-1')
        gff_copy = instance_cleanData.obtain_dicc_bed(ruta_data_bed, encoding='latin-1')
        gff_copy2 = instance_cleanData.obtain_dicc_bed(ruta_data_bed, encoding='latin-1')
        fasta = instance_cleanData.obtain_dicc_fasta(ruta_data_fasta)
        instance_cleanData.select_elements_gff(seleccionados, gff)
        list_records = instance_cleanData.extract_cds(gff, fasta)
        list_clean_records : List[Dict] = instance_cleanData.remove_sample_contaminated(list_records)
        list_clean_cds: List[Dict] = instance_cleanData.clean_cds(list_clean_records)
        
        instance_cleanData.select_elements_gff(seleccionados2, gff_copy)
        list_records = instance_cleanData.extract_sequences_mRNA(gff_copy, fasta)
        list_clean_records_mRNA : List[Dict] = instance_cleanData.remove_sample_contaminated(list_records)

        instance_cleanData.select_elements_gff(seleccionados3, gff_copy2)
        list_records = instance_cleanData.extract_sequences_counting_chr(gff_copy, fasta)
        list_clean_records : List[Dict] = instance_cleanData.remove_sample_contaminated(list_records)
        
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
        for record in data:
            seq = [vocab[nucleotide] for nucleotide in record['seq']]
            X.append(seq)
            y.append(2 if record['type'] == "UTR"
                    else 1 if record['type'] == "CDS"
                    else 0 if record['type'] == "intron"
                    else 4 if record['type'] == "transposable_element"
                    else -1) # región intergénica / elemento transponible
        
        del data
        X_list = [np.asarray(x, dtype=np.float32) for x in X]  # lista de arrays
        Y_list = [np.asarray(ya, dtype=np.float32) for ya in y]

        del X; del y

        save_chunks_to_json(X_list, Y_list, file_specie_test)

        # ---------------------------------------------------------------------------------------------

        batch_size = 128
        weights = torch.tensor([1., 1., 1., 1., 1.])
        agrupacion = 3
        instance_generateDataset  = GenerateDataset(False, agrupacion)
        device = torch.device('cuda')
        learning_rate = 1e-5 if args.fine_tunning else 5e-5
        weight_decay=1e-2
        model = TransformerClassifier(
            vocab_size=len(instance_generateDataset.vocabularyComplete)+1,
            padding_idx=len(instance_generateDataset.vocabularyComplete),
            embed_dim=128,
            num_heads=8,
            num_layers=5,
            dim_feedforward=512,
            num_classes=4,
            max_seq_len=4000,
            dropout=0.4
        ).to(device)
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)
        if not args.train:
            checkpoint = torch.load('../generate_models/model.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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

    except Exception as e:
        print("Error: ")
        print(e)
