
from .section import Section

import numpy as np

from typing import List

class Summary(Section):

    def __init__(self, html_path: str, title: str, color="#DE8512", limit_prob_gene: float = .1, limit_prob_ir: float = .3):
        super().__init__(html_path, title, color, limit_prob_ir, limit_prob_gene)
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

        different_thresholds: List[float] = [.1, .3, .5, .7, .9]

        value_gene_thresholds: List[float] = []
        value_ir_thresholds: List[float] = []
        for threshold in different_thresholds:
            value_gene_thresholds.append(super().accuracy_per_threshold(logits[mask_ir], 1, threshold))
            value_ir_thresholds.append(super().accuracy_per_threshold(logits[mask_gene], 0, threshold))
        

        html: str = ""
        html += super().create_header()
        html += super().create_section(x, y)
        x_obj, y_obj, width_obj, height_obj = x, y+self.offset_section_px, 320, 200
        html += super().add_text(f"El valor de la entropía cruzada es de {value_cross_entropy}.", x_obj, y_obj, width_obj, height_obj, image_url='assets/orange-paint-brushstroke-with-transparent-background-perfect-for-designs-and-projects-png.png')
        x_obj_2, y_obj_2, width_obj, height_obj = x_obj+500, y_obj, width_obj, height_obj
        html += super().add_text(f"El valor del accuracy es de {value_accuracy}.", x_obj_2, y_obj_2, width_obj, height_obj, image_url='assets/orange-paint-brushstroke-with-transparent-background-perfect-for-designs-and-projects-png.png')
        x_obj_3, y_obj_3, width_obj, height_obj = x_obj+1000, y_obj, width_obj, height_obj
        html += super().add_text(f"El valor del accuracy permisivo es de {value_permissive_accuracy}.", x_obj_3, y_obj_3, width_obj, height_obj, image_url='assets/orange-paint-brushstroke-with-transparent-background-perfect-for-designs-and-projects-png.png')
        x_obj_4, y_obj_4, width_obj, height_obj = x_obj+250, (y_obj+height_obj+40), width_obj, height_obj
        html += super().add_text(f"El recall de los genes es de {value_recall_gene}.", x_obj_4, y_obj_4, width_obj, height_obj, image_url='assets/orange-paint-brushstroke-with-transparent-background-perfect-for-designs-and-projects-png.png')
        x_obj_5, y_obj_5, width_obj, height_obj = x_obj+740, (y_obj+height_obj+40), width_obj, height_obj
        html += super().add_text(f"El recall de las regiones intergénicas es de {value_recall_ir}.", x_obj_5, y_obj_5, width_obj, height_obj, image_url='assets/orange-paint-brushstroke-with-transparent-background-perfect-for-designs-and-projects-png.png')
        x_obj_6, y_obj_6, width_obj, height_obj = x_obj, (y_obj+2*height_obj+100), 450, 450
        html += super().add_text(
            f"""El {value_gene_thresholds[0]}% de las regiones intergénicas poseen una probabilidad del {int(different_thresholds[0]*100)}% de contener un gen. <br> <br>
             El {value_gene_thresholds[1]}% de las regiones intergénicas poseen una probabilidad del {int(different_thresholds[1]*100)}% de contener un gen. <br> <br>
             El {value_gene_thresholds[2]}% de las regiones intergénicas poseen una probabilidad del {int(different_thresholds[2]*100)}% de contener un gen. <br> <br>
             El {value_gene_thresholds[3]}% de las regiones intergénicas poseen una probabilidad del {int(different_thresholds[3]*100)}% de contener un gen. <br> <br>
             El {value_gene_thresholds[4]}% de las regiones intergénicas poseen una probabilidad del {int(different_thresholds[4]*100)}% de contener un gen. <br> <br>
            """,
            x_obj_6,
            y_obj_6,
            width_obj,
            height_obj,
            image_url='assets/orange-and-yellow.jpg'
        )
        x_obj_7, y_obj_7, width_obj, height_obj = x_obj+700, (y_obj+2*200+100), 450, 450
        html += super().add_text(
            f"""El {value_ir_thresholds[0]}% de los genes poseen una probabilidad del {int(different_thresholds[0]*100)}% de contener una región intergénica. <br> <br>
             El {value_ir_thresholds[1]}% de los genes poseen una probabilidad del {int(different_thresholds[1]*100)}% de contener una región intergénica. <br> <br>
             El {value_ir_thresholds[2]}% de los genes poseen una probabilidad del {int(different_thresholds[2]*100)}% de contener una región intergénica. <br> <br>
             El {value_ir_thresholds[3]}% de los genes poseen una probabilidad del {int(different_thresholds[3]*100)}% de contener una región intergénica. <br> <br>
             El {value_ir_thresholds[4]}% de los genes poseen una probabilidad del {int(different_thresholds[4]*100)}% de contener una región intergénica. <br> <br>
            """,
            x_obj_7,
            y_obj_7,
            width_obj,
            height_obj,
            image_url='assets/orange-and-yellow.jpg'
        )

        super().save_html(html)