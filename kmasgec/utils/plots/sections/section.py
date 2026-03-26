import numpy as np
import io
import matplotlib.pyplot as plt
import base64
from sklearn.metrics import f1_score
import torch



class Section:

    def __init__(self, html_path: str, title: str, color, limit_prob_ir: float = .1, limit_prob_gene: float = .3):

        self.html_path = html_path
        self.title = title
        self.color = color
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.limit_prob_ir = limit_prob_ir
        self.limit_prob_gene = limit_prob_gene

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

    def add_text(self, text: str, x: int, y: int, width: int, height: int):
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

    def f1_metric(self, labels: np.array, logits: np.array):
        metrics = {}
        probs = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds = (np.asarray(probs) >= thresholds).astype(int)
        metrics['f1_micro'] = f1_score(labels, preds, average='micro')
        metrics['f1_macro'] = f1_score(labels, preds, average='macro')
        return metrics
    
    def cross_entropy(self, labels: np.array, logits: np.array):
        metrics = {}
        y_trues_t: torch.tensor = torch.tensor(labels, dtype=torch.float32)
        y_preds_t: torch.tensor = torch.tensor(logits, dtype=torch.float32)
        metrics['log_loss'] = self.loss_fn(y_trues_t, y_preds_t)
        return metrics
    
    def save_html(self, html):
        with open(self.html_path, 'a', encoding='utf-8') as f:
            f.write(html)

    def accuracy(self, logits: np.array, labels: np.array):
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
    
    def permissive_accuracy(self, logits: np.array, labels: np.array):
        # está mal. por que si alguna de las etiquetas es 0 y mi resultado tiene un 0 por ahí entonces está bien.

        probs: torch.tensor = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds: np.array = (np.asarray(probs) >= thresholds).astype(int)

        total: int = 0
        correct: int = 0

        for pred, label in zip(preds, labels):
            if np.any(pred == label):
                correct += 1
            total += 1

        return correct / total 

    def recall(self, logits: np.array, labels: np.array):
        # está mal. por que si alguna de las etiquetas es 0 y mi resultado tiene un 0 por ahí entonces está bien.

        probs: torch.tensor = torch.sigmoid(torch.tensor(logits))
        thresholds: np.array = np.array([self.limit_prob_ir, self.limit_prob_gene])
        preds: np.array = (np.asarray(probs) >= thresholds).astype(int)

        wrong: int = 0
        correct: int = 0

        for pred, label in zip(preds, labels):
            if np.any(pred == label):
                correct += 1
            else:
                wrong += 1

        return correct / (correct + wrong)