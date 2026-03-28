
from .section import Section

import numpy as np

from typing import Dict

class Summary(Section):

    def __init__(self, html_path: str, title: str, color="#39C181", limit_prob_gene: float = .1, limit_prob_ir: float = .3):
        super().__init__(html_path, title, color)
        self.limit_prob_gene = limit_prob_gene
        self.limit_prob_ir = limit_prob_ir

    def define_section(self, logits: np.array, labels: np.array, x: int, y: int):

        COLUMN_GENE: int = 1
        COLUMN_IR: int = 0
        
        value_accuracy: float = self.accuracy(logits, labels)
        value_permissive_accuracy: float = self.permissive_accuracy(logits, labels)
        value_cross_entropy: float = self.cross_entropy(logits, labels)

        mask_gene: np.array = np.all(labels == np.array([0, 1]), axis=1)
        mask_ir: np.array = np.all(labels == np.array([1, 0]), axis=1)

        value_recall_gene: float = self.recall_label(logits[mask_gene], COLUMN_GENE)
        value_recall_ir: float = self.recall_label(logits[mask_ir], COLUMN_IR)
        value_recall_micro: float = self.recall_micro(logits, labels)

        html: str = ""
        html += super().create_header()
        html += super().create_section(x, y)
        x_obj, y_obj, width_obj, height_obj = x, y+self.offset_section_px, 120, 120
        html += super().add_text(f"El valor de la entropía cruzada es de {value_cross_entropy}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El valor del accuracy es de {value_accuracy}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El valor del accuracy permisivo es de {value_permissive_accuracy}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El recall micro es de {value_recall_micro}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El recall de los genes es de {value_recall_gene}.", x_obj, y_obj, width_obj, height_obj)
        x_obj, y_obj, width_obj, height_obj = x_obj, (y_obj+height_obj), width_obj, height_obj
        html += super().add_text(f"El recall de las regiones intergénicas es de {value_recall_ir}.", x_obj, y_obj, width_obj, height_obj)

        super().save_html(html)