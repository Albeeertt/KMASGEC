
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

from typing import List, Any

class CreateGFF:

    def __init__(self, gff: pd.DataFrame, preds: List[List], places: List[List], logits: List[List]):
        self._gff = gff
        self._preds = preds
        self._places = places
        self._logits = logits

        self._change_preds = {
            0: 'intergenic_region',
            1: 'gene'
        }

    def _create_column(self, name_column: str, default_value: Any = None):
        self._gff[name_column] = default_value

    def _asArray_list(self, list_handle: List, dtype: np.typing.DTypeLike):
        return np.asarray(list_handle, dtype)
    
    def _logits_to_preds(self, list_handle: List):
        return torch.softmax(torch.tensor(list_handle), dim=1)
    
    def _add_valuesGFF(self, list_idx: List, name_column: str, value: Any):
        self._gff.loc[list_idx, name_column] = value

    def _cluster_samples(self, prob_ir: List, prob_gene: List):
        clusters_preds = defaultdict(list)
        clusters_p_ir = defaultdict(list)
        clusters_p_gene = defaultdict(list)

        for idx, pred, p_ir, p_gene in zip(self._places, self._preds, prob_ir, prob_gene):
            clusters_preds[idx].append(pred)
            clusters_p_ir[idx].append(p_ir)
            clusters_p_gene[idx].append(p_gene)

        places = []
        preds = []
        new_prob_ir = []
        new_prob_gene = []

        for key in clusters_preds.keys():
            list_preds = clusters_preds[key]
            places.append(key)
            if len(list_preds) > 1:
                list_idx_gene = np.where(list_preds == 1)[0]
                if list_idx_gene:
                    idx_key_gene = list_idx_gene[0]
                    preds.append(clusters_preds[key][idx_key_gene])
                    new_prob_ir.append(clusters_p_ir[key][idx_key_gene])
                    new_prob_gene.append(clusters_p_gene[key][idx_key_gene])
                else:
                    preds.append(clusters_preds[key][0])
                    new_prob_ir.append(clusters_p_ir[key][0])
                    new_prob_gene.append(clusters_p_gene[key][0])
            else:
                preds.append(clusters_preds[key][0])
                new_prob_ir.append(clusters_p_ir[key][0])
                new_prob_gene.append(clusters_p_gene[key][0])

        return places, preds, new_prob_ir, new_prob_gene

    
    def create_gff(self, remove_idx_mRNA: List, remove_idx_chr: List, remove_idx_startEnd: List, remove_contaminated: List, route_gff: str = None):
        
        self._create_column("Result", 'None')
        self._create_column('prob_gene', np.nan)
        self._create_column('prob_intergenic_region', np.nan)

        probs = self._logits_to_preds(self._logits)
        prob_ir = [element[0] for element in probs]
        prob_gene = [element[1] for element in probs]

        remove_idx_mRNA = self._asArray_list(remove_idx_mRNA, np.int64)
        remove_idx_chr = self._asArray_list(remove_idx_chr, np.int64)
        remove_idx_startEnd = self._asArray_list(remove_idx_startEnd, np.int64)
        remove_contaminated = self._asArray_list(remove_contaminated, np.int64)

        self._add_valuesGFF(remove_idx_mRNA, 'Result', 'No-mRNA')
        self._add_valuesGFF(remove_idx_chr, 'Result', 'No-fasta')
        self._add_valuesGFF(remove_idx_startEnd, 'Result', 'Start_bigger_than_end')
        self._add_valuesGFF(remove_contaminated, 'Result', 'Contaminated')

        places, preds, new_prob_ir, new_prob_gene = self._cluster_samples(prob_ir, prob_gene)
        
        places = self._asArray_list(places, np.int64)
        preds_names = np.vectorize(self._change_preds.get)(preds)
        prob_ir = self._asArray_list(new_prob_ir, np.float16)
        prob_gene = self._asArray_list(new_prob_gene, np.float16)

        self._add_valuesGFF(places, "Result", preds_names)
        self._add_valuesGFF(places, "prob_gene", prob_gene)
        self._add_valuesGFF(places, "prob_intergenic_region", prob_ir)


        if route_gff is not None:
            self._gff.to_csv(route_gff+"result.csv", sep=',')

        return self._gff