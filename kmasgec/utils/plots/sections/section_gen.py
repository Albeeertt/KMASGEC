from .section import Section
import numpy as np
import torch

class Gen(Section):

    def __init__(self, html_path: str, title: str, color="#6A5ACD", limit_prob_gene: float = .1, limit_prob_ir: float = .3):
        super().__init__(html_path, title, color, limit_prob_ir, limit_prob_gene)

    def define_section(self, logits : np.array, labels: np.array, x: int, y: int):

        probs = torch.sigmoid(torch.tensor(logits))
        
        mask_clean_gene = (probs[:, 0] < self.limit_prob_ir) & (probs[:, 1] >= self.limit_prob_gene)
        mask_dirty_gene = (probs[:, 0] >= self.limit_prob_ir) & (probs[:, 1] >= self.limit_prob_gene)
        mask_bad_gene = (probs[:, 1] < self.limit_prob_gene)

        gene_clean: np.array = probs[mask_clean_gene]
        annotation_gene_clean: np.array = labels[mask_clean_gene]

        gene_half: np.array = probs[mask_dirty_gene]
        annotation_gene_half: np.array = labels[mask_dirty_gene]

        gene_bad: np.array = probs[mask_bad_gene]
        annotation_gene_bad: np.array = labels[mask_bad_gene]

        correct_gene = len(gene_clean)+len(gene_half)
        dirty_gene = len(gene_half)
        bad_gene = len(gene_bad)
        metrics_f1 = super().f1_metric(labels, logits)

        metrics_logLoss_good = super().cross_entropy(gene_clean, annotation_gene_clean)
        metrics_logLoss_half = super().cross_entropy(gene_half, annotation_gene_half)
        metrics_logLoss_bad = super().cross_entropy(gene_bad, annotation_gene_bad)

        x_obj, y_obj, width_obj, height_obj = 580, 310, 120, 120

        html = ""
        html += super().create_header()
        html += super().create_section(x, y)
        html += super().add_text(f"Se han anotado correctamente {correct_gene} genes.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"De ellos, {dirty_gene} contienen algo de región intergénica.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Y {bad_gene} están anotados como gen pero el modelo no los detecta como tal.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El f1-score micro para los genes es de {metrics_f1['f1_micro']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El f1-score macro para los genes es de {metrics_f1['f1_macro']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Entropía cruzada sobre los genes sin región intergénica: {metrics_logLoss_good['log_loss']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Entropía cruzada sobre los genes con región intergénica: {metrics_logLoss_half['log_loss']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Entropía cruzada sobre los genes mal anotados: {metrics_logLoss_bad['log_loss']}.", x_obj, y_obj, width_obj, height_obj)

        super().save_html(html)




        



