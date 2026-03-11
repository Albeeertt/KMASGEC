from section import Section
import numpy as np

class Gen(Section):

    def __init__(self, html_path: str, title: str, color="#6A5ACD", limit_prob_gene: float = .1, limit_prob_ir: float = .3):
        super().__init__(html_path, title, color)
        self.limit_prob_gene = limit_prob_gene
        self.limit_prob_ir = limit_prob_ir

    def define_section(self, probs : np.array, annotations: np.array):
        
        mask_clean_gene = (probs[:, 0] < self.limit_prob_ir) & (probs[:, 1] >= self.limit_prob_gene)
        mask_dirty_gene = (probs[:, 0] >= self.limit_prob_ir) & (probs[:, 1] >= self.limit_prob_gene)
        mask_bad_gene = (probs[:, 1] < self.limit_prob_gene)

        gene_clean: np.array = probs[mask_clean_gene]
        annotation_gene_clean: np.array = annotations[mask_clean_gene]

        gene_half: np.array = probs[mask_dirty_gene]
        annotation_gene_half: np.array = annotations[mask_dirty_gene]

        gene_bad: np.array = probs[mask_bad_gene]
        annotation_gene_bad: np.array = annotations[mask_bad_gene]

        correct_gene = len(gene_clean)+len(gene_half)
        dirty_gene = len(gene_half)
        bad_gene = len(gene_bad)

        metrics_f1 = super().f1_metrics(probs, annotations)

        metrics_logLoss_good = super().cross_entropy(gene_clean, annotation_gene_clean)
        metrics_logLoss_half = super().cross_entropy(gene_half, annotation_gene_half)
        metrics_logLoss_bad = super().cross_entropy(gene_bad, annotation_gene_bad)

        html = ""
        html += super().create_section()
        html += super().add_text(f"Se han anotado correctamente {correct_gene} genes.", 20, 20, 60, 60)
        html += super().add_text(f"De ellos, {dirty_gene} contienen algo de región intergénica.", 20, 40, 60, 60)
        html += super().add_text(f"Y {bad_gene} están anotados como gen pero el modelo no los detecta como tal.", 20, 60, 60, 60)
        html += super().add_text(f"El f1-score micro para los genes es de {metrics_f1['f1_micro']}.", 20, 80, 60, 60)
        html += super().add_text(f"El f1-score macro para los genes es de {metrics_f1['f1_macro']}.", 20, 100, 60, 60)
        html += super().add_text(f"Entropía cruzada sobre los genes sin región intergénica: {metrics_logLoss_good['log_loss']}.", 20, 120, 60, 60)
        html += super().add_text(f"Entropía cruzada sobre los genes con región intergénica: {metrics_logLoss_half['log_loss']}.", 20, 140, 60, 60)
        html += super().add_text(f"Entropía cruzada sobre los genes mal anotados: {metrics_logLoss_bad['log_loss']}.", 20, 160, 60, 60)

        super().save_html(html)




        



