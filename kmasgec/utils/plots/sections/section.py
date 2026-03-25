import numpy as np
import io
import matplotlib.pyplot as plt
import base64
from sklearn.metrics import f1_score, log_loss



class Section:

    def __init__(self, html_path: str, title: str, color):

        self.html_path = html_path
        self.title = title
        self.color = color

    def create_section(self):
        html = f"""
        <section style="
            padding:50px;
            text-align:center;
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
                💛
            </p>
        </section>
        """
        return html

    def add_text(self, text: str, x: int, y: int, width: int, height: int):
        html = f"""
        <div style="
            position:relative;
            width:{width}px;
            height:{height}px;
            border:1px dashed red;
        ">
            <div style="
                position:absolute;
                left:{x}px;
                top:{y}px;
                font-size:12px;
                color:{self.color};
                overflow:hidden;
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

    def end_section(self, y: int, width: int):
        html = f"""
        <div style="
            position:absolute;
            left:0px;
            top:{y}px;
            width:{width}px;
            text-align:center;
            color:#555;
            font-size:18px;
        ">
            ───────── 💛 ─────────
        </div>
        """
        return html

    def define_section(self):
        pass

    def f1_metric(y_trues: np.array, y_preds: np.array):
        metrics = {}
        metrics['f1_micro'] = f1_score(y_trues, y_preds, average='micro')
        metrics['f1_macro'] = f1_score(y_trues, y_preds, average='macro')
        return metrics
    
    def cross_entropy(y_trues: np.array, y_preds: np.array):
        metrics = {}
        metrics['log_loss'] = log_loss(y_trues, y_preds)
        return metrics
    
    def save_html(self, html):
        with open(self.html_path, 'a', encoding='utf-8') as f:
            f.write(html)