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
from kmasgec.core.models.model_architecture.transformers import TransformerClassifier_attnPool
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
    parser.add_argument('--add_labels', action='store_true', help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--fine_tunning', action='store_true', help="")
    parser.add_argument('--train', action='store_true', help="Si deseas entrenar un modelo desde cero")
    parser.add_argument('--gpus', type=str, default="", help="GPUs a usar, e.g. '0', '0,1', '0,2,3'", required=True) # TODO: ignorar, hacer un único parser y ya.
    parser.add_argument("--lens_mode", action="store_true", help="Divide las secuencias en trozos.")
    parser.add_argument("--zoom_length", type=int, required=False, help="Tamaño de las subsecuencias.")

    # Analizar los argumentos pasados por el usuario
    return parser.parse_args()


def ejecutar():
    MAX_LEN_SEQ = 10000 
    NAME_HTML: str = 'info.html'

    agrupacion = 3
    kmer: bool = True

    args = obtener_argumentos()

    route_out = args.out
    if not os.path.exists(route_out):
        os.mkdir(route_out)
        
    route_out = route_out+'/' if not route_out.endswith('/') else route_out

    if args.lens_mode and not args.zoom_length:
        print("arg. zoom_length is necessary with lens_mode.")
        return
        
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
        dataframe_elements_plus_te_mRNA = instance_modify_samples.lends_mode(dataframe_elements_plus_te_mRNA, MAX_LEN_SEQ, args.zoom_length)
    fasta = instance_cleanData.obtain_dicc_fasta(ruta_data_fasta)

    # First Data
    # ---------------------------------------------------------------------------------------------


    data_first_algorithm = dataframe_elements_plus_te_mRNA[dataframe_elements_plus_te_mRNA['type'].isin(['intergenic_region', 'gene'])].copy()

    data_first_algorithm[['start','end']] = data_first_algorithm[['start','end']].apply(pd.to_numeric, errors='coerce')
    
    # list_records, remove_idx_chr, remove_idx_startEnd = instance_cleanData.extract_sequences_counting_chr(data_first_algorithm, fasta)
    # list_clean_records, remove_contaminated = instance_cleanData.remove_sample_contaminated(list_records)

    # vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # X = []
    # y = []
    # place = []
    # for record in list_clean_records:
    #     seq = [vocab[nucleotide] for nucleotide in record['seq']]
    #     X.append(seq)
    #     y.append(np.array(1) if record['type'] == "gene"
    #         else np.array(0) if record['type'] == "intergenic_region"
    #         else -1) # región intergénica / elemento transponible
    #     place.append(record['old_idx'])

    # X_fin = [np.asarray(i, dtype=np.float32) for i in X]
    # y_fin = [np.asarray(i, dtype=np.float32) for i in y]
    # place_fin = [np.asarray(i, dtype=np.int64) for i in place]
    # save_all_to_json(X_fin, y_fin, place_fin, filename=ruta_data_first_algorithm, names=['X', 'Y', 'Place'])

    remove_idx_chr = []
    remove_idx_startEnd = []
    remove_contaminated = []
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for sample in data_first_algorithm.to_dict(orient='records'):
        new_record, add_idx_chr, add_idx_startEnd = instance_cleanData.extract_sample_counting_chr(sample, fasta)
        if new_record is None:
            remove_idx_chr.extend(add_idx_chr)
            remove_idx_startEnd.extend(add_idx_startEnd)
            continue
        if instance_cleanData.is_contaminated(new_record):
            remove_contaminated.append(new_record['old_idx'])
            continue
        X = []
        y = []
        place = []
        seq = [vocab[nucleotide] for nucleotide in new_record['seq']]
        X.append(seq)
        y.append(np.array(1) if new_record['type'] == "gene"
            else np.array(0) if new_record['type'] == "intergenic_region"
            else -1) # región intergénica / elemento transponible
        place.append(new_record['old_idx'])

        X_fin = [np.asarray(i, dtype=np.float32) for i in X]
        y_fin = [np.asarray(i, dtype=np.float32) for i in y]
        place_fin = [np.asarray(i, dtype=np.int64) for i in place]

        save_all_to_json(X_fin, y_fin, place_fin, filename=ruta_data_first_algorithm, names=['X', 'Y', 'Place'])


    # Model 1
    # ---------------------------------------------------------------------------------------------

    batch_size: int = args.batch_size
    min_len_seq: Dict[int, int] = {0: 10, 1: 10, 2: 10, 3: 10}
    instance_generateDataset  = GenerateDataset(False, agrupacion, kmer)
    padding_value = len(instance_generateDataset.vocabularyComplete)
    vocab_size = len(instance_generateDataset.vocabularyComplete)+1
    print("Tamaño del vocabulario: ", len(instance_generateDataset.vocabularyComplete))
    partial_collateFN = partial(collate_fn_oneHead, padding_value=padding_value)

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
        embed_dim=512, 
        num_heads=8,
        num_layers=8, 
        dim_feedforward=4096, 
        num_classes=2, 
        dropout=0.2
    )
    torch.compile(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss()

    checkpoint = torch.load(pkg_resources.resource_filename("kmasgec", "generate_models/cnn_simple_yesAttnMask_transformer_2.pt"), map_location=device)
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
    report_dict, all_trues, all_preds, all_places, all_softmax_official_values = iteration_test_oneHead(pbar_test,  model, device, criterion, 2)
    pbar_test.close()
    os.remove(ruta_data_first_algorithm)

    model.to('cpu')
    del model
    
    gc.collect()
    if device_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    instance_createGFF = CreateGFF(gff, all_preds, all_places, all_softmax_official_values)
    instance_createGFF.create_gff(remove_idx_mRNA, remove_idx_chr, remove_idx_startEnd, remove_contaminated, route_out)

    with open(route_out+'report.json', "a") as f:
        json.dump(report_dict, f, indent=4)

    # gff['Result'] = 'None'
    # gff['prob_gene'] = np.nan
    # gff['prob_intergenic_region'] = np.nan

    # a_places = np.asarray(all_places, dtype=np.int64)
    # a_preds  = np.asarray(all_preds, dtype=int)
    # a_trues  = np.asarray(all_trues, dtype=int)
    # probs = torch.softmax(torch.tensor(all_softmax_official_values), dim=1)
    # probs_ir = np.asarray([ element[0] for element in probs], dtype=np.float16)
    # probs_gene = np.asarray([ element[1] for element in probs], dtype=np.float16)
    # # probs_gene_ir = np.asarray([ element[1] for element in probs], dtype=np.float16)
    # a_remove_idx_mRNA = np.asarray(remove_idx_mRNA, dtype=np.int64)
    # a_remove_idx_chr = np.asarray(remove_idx_chr, dtype=np.int64)
    # a_remove_idx_startEnd = np.asarray(remove_idx_startEnd, dtype=np.int64)
    # a_remove_contaminated = np.asarray(remove_contaminated, dtype=np.int64)

    # # preds = (a_preds > 0.5).astype(int)
    # # codes = preds[:, 0] * 2 + preds[:, 1]
    # # mapping = {
    # # 1: 'gen',
    # # 2: 'región intergénica',
    # # 3: 'gen_into_ri',
    # # 0: 'ninguno'
    # # }

    # label_map = {
    #     0: "región intergénica",
    #     1: "gen_ir",
    #     2: "gen"
    # }

    # labels = np.vectorize(label_map.get)(a_preds)

    # # labels = np.where(np.isin(a_preds, [0, 1]), 'gen', 'región intergénica')

    # gff.loc[a_places, 'Result'] = labels
    # gff.loc[a_remove_idx_mRNA, 'Result'] = 'No-mRNA'
    # # gff.loc[a_remove_idx_mRNA, 'Bad'] = 'Not-considered'

    # gff.loc[a_remove_idx_chr, 'Result'] = 'No-fasta'
    # # gff.loc[a_remove_idx_chr, 'Bad'] = 'Not-considered'

    # gff.loc[a_remove_idx_startEnd, 'Result'] = 'Start_bigger_than_end'
    # # gff.loc[a_remove_idx_startEnd, 'Bad'] = 'Not-considered'

    # gff.loc[a_remove_contaminated, 'Result'] = 'Contaminated'
    # # gff.loc[a_remove_contaminated, 'Bad'] = 'Not-considered'

    # gff.loc[a_places, 'prob_gene'] = probs_gene
    # gff.loc[a_places, 'prob_intergenic_region'] = probs_ir
    # # gff.loc[a_places, 'prob_gen_ri'] = probs_gene_ir

    # # bad_mask = a_trues != a_preds
    # # bad_idx  = a_places[bad_mask] 
    # # gff.loc[bad_idx, 'Bad'] = 'Yes'

    # gff.to_csv(route_out+'result.csv', sep=',')

    # LIMIT_PROB_GENE, LIMIT_PROB_IR =.5, .5

    # instance_html_gen = Gen(args.html_path, "Desglose", LIMIT_PROB_GENE, LIMIT_PROB_IR)
    # instance_html_ir = IntergenicRegion(args.html_path, 'ir ir ir', LIMIT_PROB_GENE, LIMIT_PROB_IR)
    # instance_html_summary = Summary(route_out+NAME_HTML, 'Summary', "#DE8512", LIMIT_PROB_GENE, LIMIT_PROB_IR)


    # mask_gene = np.all(a_trues == [0, 1], axis=1)
    # mask_ir = np.all(a_trues == [1, 0], axis=1)

    # all_softmax_official_values: np.array = np.asarray(all_softmax_official_values, dtype=float)

    # instance_html_summary.define_section(all_softmax_official_values, a_trues, 200, 0)

    # instance_html_gen.define_section(all_softmax_official_values[mask_gene], a_trues[mask_gene], 200, 1260)
    # instance_html_ir.define_section(all_softmax_official_values[mask_ir], a_trues[mask_ir], 200, 2520)


    # instance_html_gen = Gen('./prueba.html', "gen gen gen")
    # instance_html_ir = IntergenicRegion('./prueba.html', 'IR IR IR')

    # mask_gene = np.all(a_trues == [0, 1], axis=1)
    # mask_ir = np.all(a_trues == [1, 0], axis=1)

    # instance_html_gen.define_section(a_preds[mask_gene], a_trues[mask_gene])
    # instance_html_ir.define_section(a_preds[mask_ir], a_trues[mask_ir])