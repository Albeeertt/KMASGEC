from .section import Section
import numpy as np
import torch

class IntergenicRegion(Section):

    def __init__(self, html_path: str, title: str, color="#CEA439", limit_prob_gene: float = .1, limit_prob_ir: float = .3):
        super().__init__(html_path, title, color, limit_prob_ir, limit_prob_gene)

    def define_section(self, logits: np.array, labels: np.array, x: int, y: int):

        probs = torch.sigmoid(torch.tensor(logits))

        mask_clean_ir = (probs[:, 0] >= self.limit_prob_ir) & (probs[:, 1] < self.limit_prob_gene)
        mask_dirty_ir = (probs[:, 0] >= self.limit_prob_ir) & (probs[:, 1] >= self.limit_prob_gene)
        mask_bad_ir = (probs[:, 0] < self.limit_prob_ir) & (probs[:, 1] >= self.limit_prob_gene)

        ir_clean: np.array = probs[mask_clean_ir]
        annotation_ir_clean: np.array = labels[mask_clean_ir]

        ir_half: np.array = probs[mask_dirty_ir]
        annotation_ir_half: np.array = labels[mask_dirty_ir]

        ir_bad: np.array = probs[mask_bad_ir]
        annotation_ir_bad: np.array = labels[mask_bad_ir]

        correct_ir = len(ir_clean)+len(ir_half)
        dirty_ir = len(ir_half)
        bad_ir = len(ir_bad)

        metrics_f1 = super().f1_metric(labels, logits)

        metrics_logLoss_good = super().cross_entropy(ir_clean, annotation_ir_clean)
        metrics_logLoss_half = super().cross_entropy(ir_half, annotation_ir_half)
        metrics_logLoss_bad = super().cross_entropy(ir_bad, annotation_ir_bad)

        html = ""
        html += super().create_section(x, y)
        x_obj, y_obj, width_obj, height_obj = 980, 310, 120, 120
        html += super().add_text(f"Se han anotado correctamente {correct_ir} regiones intergénicas.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"De ellos, {dirty_ir} contienen algo de gen.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Y {bad_ir} están anotados como regiones intergénicas pero el modelo no los detecta como tal.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El f1-score micro para las regiones intergénicas es de {metrics_f1['f1_micro']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El f1-score macro para las regiones intergénicas es de {metrics_f1['f1_macro']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Entropía cruzada sobre las regiones intergénicas sin genes: {metrics_logLoss_good['log_loss']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Entropía cruzada sobre las regiones intergénicas con genes: {metrics_logLoss_half['log_loss']}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"Entropía cruzada sobre las regiones intergénicas mal anotadas: {metrics_logLoss_bad['log_loss']}.", x_obj, y_obj, width_obj, height_obj)

        super().save_html(html)