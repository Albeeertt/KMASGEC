import numpy as np
import io
import matplotlib.pyplot as plt
import base64
from sklearn.metrics import f1_score
import torch

from typing import Dict


class Section:

    def __init__(self, html_path: str, title: str, color, limit_prob_ir: float = .1, limit_prob_gene: float = .3):

        self.html_path = html_path
        self.title = title
        self.color = color
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.limit_prob_ir = limit_prob_ir
        self.limit_prob_gene = limit_prob_gene
        self.offset_section_px: int = 310

    def create_header(self):
        html = """
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <title>Prueba</title>
            </head>
        """

        return html

    def create_section(self, x: int, y: int):
        html = f"""
        <section style="
            position:absolute;
            top:{y}px;
            left:{x}px;
            width:1400px;
            height:260px;
            padding:50px;
            text-align:center;
            box-sizing:border-box;
            background:#f5f5f5;
            border-radius:10px;
            margin:30px 0;
        ">
            <h2 style="
                color:{self.color};
                font-size:2.6em;
                font-weight:bold;
                letter-spacing:1px;
            ">
                {self.title}
            </h2>

            <p style="
                color:{self.color};
                font-size:1.2em;
            ">
                🫨
            </p>
        </section>
        """
        return html

    def add_text(self, text: str, x: int, y: int, width: int, height: int) -> str:
        html = f"""
        <div style="
            position:absolute;
            width:{width}px;
            height:{height}px;
            left:{x}px;
            top:{y}px;
            border:1px dashed red;
            margin:0 auto;
        ">
            <div style="
                position:relative;
                font-size:12px;
                color:{self.color};
                background:rgba(0,0,255,0.1); /* debug */
            ">
                {text}
            </div>
        </div>
        """
        return html

    def add_graph(self, fig, x:int, y:int, width: int, height: int):
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)

        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        html = f"""
            <img src="data:image/png;base64,{img_base64}" style="
                position:absolute;
                left:{x}px;
                top:{y}px;
                width:{width}px;
                height:{height}px;
                border: 1px dashed red;
            ">
            """
        return html

    def define_section(self):
        pass

    def f1_metric(self, logits: np.array, labels: np.array) -> Dict:
        metrics = {}
        probs = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds = (np.asarray(probs) >= thresholds).astype(int)
        metrics['f1_micro'] = f1_score(labels, preds, average='micro')
        metrics['f1_macro'] = f1_score(labels, preds, average='macro')
        return metrics
    
    def cross_entropy(self, logits: np.array, labels: np.array) -> float:
        y_trues_t: torch.tensor = torch.tensor(labels, dtype=torch.float32)
        y_preds_t: torch.tensor = torch.tensor(logits, dtype=torch.float32)
        value_cross_entropy: torch.tensor = self.loss_fn(y_preds_t, y_trues_t)
        return value_cross_entropy.item()
    
    def save_html(self, html):
        with open(self.html_path, 'a', encoding='utf-8') as f:
            f.write(html)

    def accuracy(self, logits: np.array, labels: np.array) -> float:
        # TODO: me gustaría tener un accuracy estricto y otro más permisivo. con distintos thresholds creo que puede llegar a ser interesante.
        # Si la entropía cruzada es baja los thresholds pueden ser más estrictos y viceversa, pero eso es como hacer trampas.
        # Observar los thresholds poco a poco. De momento, centrarse en sacar varias cosas para que esto sea útil para sacar resultados.
        # Mostrar accuracy con varios thresholds sí.

        probs: torch.tensor = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds: np.array = (np.asarray(probs) >= thresholds).astype(int)

        total: int = 0
        correct: int = 0

        for pred, label in zip(preds, labels):
            if np.all(pred == label):
                correct += 1
            total += 1

        return correct / total
    
    def permissive_accuracy(self, logits: np.array, labels: np.array) -> float:

        probs: torch.tensor = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds: np.array = (np.asarray(probs) >= thresholds).astype(int)

        total: int = 0
        correct: int = 0

        for pred, label in zip(preds, labels):
            if not np.array_equal(pred, np.array([0, 0])) and np.any(pred == label):
                correct += 1
            total += 1
        return correct / total 

    def recall_label(self, logits: np.array, column: int) -> float:
        # Los valores que llegan deben de estar filtrados por la etiqueta de la que se quiere obtener el recall.

        probs: torch.tensor = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds: np.array = (np.asarray(probs) >= thresholds).astype(int)

        correct: int = 0

        for pred in preds:
            if pred[column] == 1:
                correct += 1

        return correct / len(preds)
    
    def recall_micro(self, logits: np.array, labels: np.array) -> float:
        # este recall no debe de ser filtrado, no como recall_label, la intuición de este es hacer un recall binario al uso gigante, por tanto, se aplanan los logits y labels.
        # El significado detrás de este valor es: cuántos 1s es mi modelo capaz de detectar.
        # Lo que significa, cuántas regiones intergénicas y genes en mi modelo capaz de detectar.

        probs: torch.tensor = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds: np.array = (np.asarray(probs) >= thresholds).astype(int)

        TP = np.sum((labels == 1) & (preds == 1))
        FN = np.sum((labels == 1) & (preds == 0))

        return TP / (TP + FN) if (TP + FN) > 0 else 0.0