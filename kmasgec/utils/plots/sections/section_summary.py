
from .section import Section

import numpy as np

class Summary(Section):

    def __init__(self, html_path: str, title: str, color="#39C181", limit_prob_gene: float = .1, limit_prob_ir: float = .3):
        super().__init__(html_path, title, color)
        self.limit_prob_gene = limit_prob_gene
        self.limit_prob_ir = limit_prob_ir

    def define_section(self, logits: np.array, labels: np.array, x: int, y: int):
        pass