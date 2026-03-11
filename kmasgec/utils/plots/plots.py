import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

class Plots:

    def pie_chart(self, df, column, title='Elementos por clase (type)',
                explode_labels=None, explode_amount=0.12, cmap='tab20'):
        """
        explode_labels: conjunto/lista de etiquetas a separar (opcional)
        """
        plt.style.use('ggplot')
        legend_fmt='{label} (n={count})'

        counts = df[column].value_counts()
        counts = counts.sort_values(ascending=False)
        labels = [str(x) for x in counts.index.tolist()]
        sizes  = counts.values
        total  = sizes.sum()
        n = len(labels)

        # colores automáticos
        cmap_obj = plt.cm.get_cmap(cmap, n)
        colors = [cmap_obj(i) for i in range(n)]

        # explode solo para ciertas etiquetas
        explode_labels = set(map(str, explode_labels)) if explode_labels else set()
        explode = [explode_amount if lbl in explode_labels else 0 for lbl in labels]


        legend_labels = [
        legend_fmt.format(label=lbl, count=int(cnt), pct=(cnt/total))
        for lbl, cnt in zip(labels, sizes)
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.subplots_adjust(right=0.70)  # hueco para la leyenda

        wedges, _, _ = ax.pie(
            sizes,
            colors=colors,
            explode=explode,
            startangle=90,
            autopct='%1.1f%%',
            pctdistance=0.8
        )
        ax.axis('equal')
        ax.set_title(title)

        ax.legend(
            wedges, legend_labels,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            title='Clase'
        )
        return fig, ax


    def resumen_comparativa(self, df_before, df_after, column='type'):
        c0 = df_before[column].value_counts()
        c1 = df_after[column].value_counts()
        clases = sorted(set(c0.index).union(c1.index))
        before = c0.reindex(clases, fill_value=0)
        after  = c1.reindex(clases, fill_value=0)
        summary = pd.DataFrame({
            'class': clases,
            'before': before.values,
            'after':  after.values
        })
        summary['eliminated'] = (summary['before'] - summary['after']).clip(lower=0)
        summary['pct_eliminated'] = np.where(summary['before']>0,
                                            summary['eliminated']/summary['before'],
                                            0.0)
        return summary

    def plot_barras_agrupadas(self, summary, title='Comparativa eliminados por clase'):
        labels = summary['class'].tolist()
        x = np.arange(len(labels))
        w = 0.38

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, summary['before'], width=w, label='Antes')
        ax.bar(x + w/2, summary['after'],  width=w, label='Ahora')

        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_ylabel('N')
        ax.set_title(title)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.02))
        fig.tight_layout()
        return fig, ax

    def plot_eliminados(self, summary, title='Eliminados por clase'):
        s = summary.sort_values('eliminated', ascending=True)
        fig, ax = plt.subplots(figsize=(9, 0.45*len(s) + 1))
        ax.barh(s['class'], s['eliminated'])
        # Etiquetas n (pct)
        for i, (n_elim, pct) in enumerate(zip(s['eliminated'], s['pct_eliminated'])):
            ax.text(n_elim, i, f' {int(n_elim)} ({pct*100:.1f}%)', va='center')
        ax.set_xlabel('N eliminados')
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax
    

    def bar_binned_lengths_by_class(
        self,
        df: pd.DataFrame,
        class_col: str = "type",
        start_col: str = "start",
        end_col: str = "end",
        bin_size: int = 200,
        cap: int | None = None,   # hasta este límite se crean bins regulares; lo que supere va a "≥cap"
    ):
        df["length"] = df[end_col] - df[start_col] + 1

        # 2) construir edges
        if cap is not None:
            # alineamos cap al múltiplo de bin_size superior para bins uniformes
            cap_aligned = ((cap + bin_size - 1) // bin_size) * bin_size
            Lmax = int(df["length"].max())
            if Lmax <= cap_aligned:
                edges = list(range(0, cap_aligned, bin_size)) + [cap_aligned]
            else:
                edges = list(range(0, cap_aligned, bin_size)) + [cap_aligned, float("inf")]
        else:
            # sin cap: como antes, hasta el máximo y un bin final "≥Lmax"
            Lmax = int(df["length"].max())
            Lmax = ((Lmax + bin_size - 1) // bin_size) * bin_size
            edges = list(range(0, Lmax, bin_size)) + [Lmax, float("inf")]

        # 3) cortar en bins [a,b)
        df["bin"] = pd.cut(df["length"], bins=edges, right=False, include_lowest=True)

        # 4) tabla bin x clase
        counts = df.groupby(["bin", class_col]).size().unstack(fill_value=0)

        # 5) plot
        ax = counts.plot(kind="bar", figsize=(10, 4.5), rot=45)
        ax.set_xlabel("Longitud (bins)")
        ax.set_ylabel("n muestras")

        # 6) etiquetas: "0–200", "200–400", "≥1000"
        if len(counts.index) > 0:
            labels = []
            for iv in counts.index:
                left = int(iv.left) if np.isfinite(iv.left) else iv.left
                right = iv.right
                labels.append(f"≥{left}" if np.isinf(right) else f"{left}–{int(right)}")
            ax.set_xticklabels(labels)

        fig = ax.figure           # <- aquí obtienes la Figure
        fig.tight_layout()

        return fig, ax
    


    def bar_binned(
        self,
        df: pd.DataFrame,
        class_col: str = "type",
        value_col: str = "start",
        bin_size: float = 1.0,
        cap: float | None = None,   # hasta este límite se crean bins regulares; lo que supere va a "≥cap"
    ):
        # 2) construir edges
        if cap is not None:
            # alineamos cap al múltiplo de bin_size superior para bins uniformes
            cap_aligned = np.ceil(cap / bin_size) * bin_size
            Lmax = float(df[value_col].max())
            if Lmax <= cap_aligned:
                edges = np.arange(0, cap_aligned + 1e-12, bin_size).tolist()
            else:
                edges = np.arange(0, cap_aligned + 1e-12, bin_size).tolist() + [float("inf")]
        else:
            # sin cap: hasta el máximo y un bin final "≥Lmax"
            Lmax = float(df[value_col].max())
            Lmax = np.ceil(Lmax / bin_size) * bin_size
            edges = np.arange(0, Lmax + 1e-12, bin_size).tolist() + [float("inf")]

        # 3) cortar en bins [a,b)
        df = df.copy()
        df["bin"] = pd.cut(df[value_col], bins=edges, right=False, include_lowest=True)

        # 4) tabla bin x clase
        counts = df.groupby(["bin", class_col]).size().unstack(fill_value=0)
        counts = counts.sort_index()  # <- para asegurar orden correcto en el eje X

        # 5) plot
        ax = counts.plot(kind="bar", figsize=(10, 4.5), rot=45)
        ax.set_xlabel(f"{value_col} (bins)")
        ax.set_ylabel("n muestras")

        # 👇 fija posiciones antes de poner etiquetas
        ax.set_xticks(np.arange(len(counts.index)))

        # 6) etiquetas: "0.0–0.1", ..., "≥1.0"
        if len(counts.index) > 0:
            labels = []
            for iv in counts.index:
                left, right = iv.left, iv.right
                if np.isinf(right):
                    labels.append(f"≥{left:.1f}")
                else:
                    labels.append(f"{left:.1f}–{right:.1f}")
            ax.set_xticklabels(labels)

        fig = ax.figure
        fig.tight_layout()
        return fig, ax


    def compute_total_at_cg(self, df: pd.DataFrame, class_col="type", seq_col="seq"):
        s = df[seq_col].fillna("").astype(str).str.upper()
        A = s.str.count("A"); C = s.str.count("C"); G = s.str.count("G"); T = s.str.count("T")

        agg = (pd.DataFrame({class_col: df[class_col], "A":A, "C":C, "G":G, "T":T})
                .groupby(class_col, sort=False)[["A","C","G","T"]].sum()
                .reset_index())

        agg["AT_count"] = agg["A"] + agg["T"]
        agg["CG_count"] = agg["C"] + agg["G"]
        agg["total"]    = agg["AT_count"] + agg["CG_count"]
        agg["AT_pct"]   = np.where(agg["total"]>0, agg["AT_count"]/agg["total"]*100, 0.0)
        agg["CG_pct"]   = 100.0 - agg["AT_pct"]
        fig, ax = plt.subplots(figsize=(8, 0.4*len(agg) + 1))
        ax.axis("off")
        tbl = ax.table(cellText=agg.round(2).values,
                    colLabels=agg.columns.tolist(),
                    loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        fig.tight_layout()
        return fig, ax
    


    def stats_by_class(
        self,
        df: pd.DataFrame,
        class_col: str = "type",
        value_col: str = "seq",        
        round_to: int = 3            
    ) -> pd.DataFrame:
        # Asegura numérico

        df[value_col] = (df["seq"].str.len())
        s = pd.to_numeric(df[value_col], errors="coerce")

        work = pd.DataFrame({class_col: df[class_col].values, value_col: s.values})

        out = (work
            .groupby(class_col, sort=False)
            .agg(
                n=(value_col, "size"),
                mean=(value_col, "mean"),
                median=(value_col, "median"),
                std=(value_col, "std"),     # ← añadido
                min=(value_col, "min"),
                max=(value_col, "max"),
                p50=(value_col, lambda x: x.quantile(0.50)),
                p75=(value_col, lambda x: x.quantile(0.75)),
                p90=(value_col, lambda x: x.quantile(0.90)),
                p95=(value_col, lambda x: x.quantile(0.95)),
                p97=(value_col, lambda x: x.quantile(0.97)) 

            )
            .reset_index()
        )

        out[["mean","median","std","min","max","p50","p75","p90", "p95", "p97"]] = \
            out[["mean","median","std","min","max","p50","p75","p90", "p95", "p97"]].round(round_to)

        fig, ax = plt.subplots(figsize=(8, 0.4*len(out) + 1))
        ax.axis("off")
        tbl = ax.table(cellText=out.round(2).values,
                    colLabels=out.columns.tolist(),
                    loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        fig.tight_layout()
        return fig, ax



    def init_report(self, html_path: str, assets_path: str, main_title: str, subtitle: str | None = None):
        """
        Crea un HTML con estilo básico y un <main> donde iremos añadiendo secciones.
        Si ya existe, no lo pisa.
        """
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(assets_path):
            os.makedirs(assets_path)
        if html_path.exists():
            return str(html_path), str(assets_path)

        style = """
        <style>
        :root { color-scheme: light dark; }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; line-height: 1.55; }
        header { margin-bottom: 24px; text-align: center; }
        h1 { font-size: 2rem; margin: 0 0 6px; }
        .subtitle { opacity: .8; margin-bottom: 8px; }
        section { margin: 28px 0; }
        .card { padding: 16px; border-radius: 14px; box-shadow: 0 4px 16px rgba(0,0,0,.08); }
        img { width: 100%; height: auto; border-radius: 10px; }
        .muted { opacity: .75; }

        /* Títulos y texto de sección centrados */
        .section-title { text-align: center; margin: 0 0 10px; }
        .section-text  { text-align: center; }

        /* Separador entre secciones (línea sutil) */
        .section-hr { border: 0; border-top: 1px solid rgba(127,127,127,.35); margin: 22px 0; }

        /* Separador mini entre título y gráficas (no es línea) */
        .mini-divider { text-align: center; margin: 6px 0 14px; opacity: .5; font-size: 16px; }
        .mini-divider::before { content: "• • •"; letter-spacing: 6px; }
        </style>
        """.strip()

        skeleton = f"""<!doctype html>
                    <html lang="es">
                    <head>
                    <meta charset="utf-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1" />
                    <title>{main_title}</title>
                    {style}
                    </head>
                    <body>
                    <header>
                        <h1>{main_title}</h1>
                        <div class="subtitle muted">{subtitle or ''}</div>
                    </header>
                    <main id="content">
                    </main>
                    </body>
                    </html>
                    """
        html_path.write_text(skeleton, encoding="utf-8")
        return str(html_path), str(assets_path)



    def add_plot_section(self, html_path: str, assets_path: str, fig, section_title: str,
                        img_name: str | None = None,
                        dpi: int = 160, extra_text: str | None = None):
        """
        - Si `section_title` tiene texto -> crea NUEVA sección (título fuera, centrado) y añade <hr> gruesa solo en ese caso.
        - Si `section_title` es None/""/espacios o 'none'/'null' -> NO crea sección; añade la imagen a la ÚLTIMA sección existente.
        """
        html_path = Path(html_path)
        if not html_path.exists():
            raise FileNotFoundError(f"El HTML no existe: {html_path}")

        assets_dir = Path(assets_path)
        assets_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(fig, "savefig"):
            raise TypeError("`fig` debe ser una Figure de matplotlib.")

        # Quitar títulos internos de la figura (el título va fuera, en HTML)
        try:
            if getattr(fig, "_suptitle", None) is not None:
                try: fig._suptitle.remove()
                except Exception: fig.suptitle("")
            else:
                fig.suptitle("")
        except Exception:
            pass
        for ax in getattr(fig, "axes", []):
            try: ax.set_title("")
            except Exception: pass

        # Normalizar título
        raw = section_title
        title_norm = ("" if raw is None else str(raw)).strip()
        is_new_section = title_norm.lower() not in {"", "none", "null", "nil"}

        # Guardar imagen
        stub = "".join(c if c.isalnum() or c in "-_" else "_" for c in title_norm)[:40] or "grafica"
        img_name = img_name or f"{stub}.png"
        img_path = assets_dir / img_name
        fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        rel_img_src = os.path.relpath(img_path, start=html_path.parent)
        html = html_path.read_text(encoding="utf-8")

        # Estilos inline (sin CSS externo)
        hr_style   = 'style="border:0;border-top:3px solid rgba(127,127,127,.65);margin:26px 0"'  # más gruesa
        grid_style = 'display:flex;flex-wrap:wrap;gap:12px;justify-content:center;align-items:flex-start'
        img_style  = 'width:42%;max-width:520px;min-width:260px;flex:1 1 320px;height:auto;display:block;margin:0'

        if is_new_section:
            # <hr> solo si ya había secciones
            had_sections = "<section" in html
            hr_html = f'<hr {hr_style} />\n' if had_sections else ""
            section_html = f"""
    {hr_html}<section class="card">
        <h2 class="section-title" style="text-align:center;margin:0 0 12px;">{title_norm}</h2>
        {f'<p class="muted section-text" style="text-align:center;margin:0 0 12px;">{extra_text}</p>' if extra_text else ''}
        <div class="plot-grid" style="{grid_style}">
        <img src="{rel_img_src}" alt="{title_norm}" style="{img_style}">
        <!-- PLOTS-INLINE-ANCHOR -->
        </div>
    </section>
    """
            if "</main>" not in html:
                raise ValueError("El HTML no contiene <main id='content'>…</main>")
            html = html.replace("</main>", section_html + "  </main>")
        else:
            # Añadir a la ÚLTIMA sección (si hay contenedor, metemos dentro; si no, fallback antes de </section>)
            anchor = "<!-- PLOTS-INLINE-ANCHOR -->"
            pos = html.rfind(anchor)
            img_tag = f'<img src="{rel_img_src}" alt="grafica" style="{img_style}">\n      '
            if pos != -1:
                html = html[:pos] + img_tag + html[pos:]
            else:
                # Fallback para secciones antiguas sin contenedor
                end_tag = "</section>"
                pos2 = html.rfind(end_tag)
                if pos2 == -1:
                    raise ValueError("No hay ninguna sección aún. Crea primero una pasando `section_title`.")
                # Creamos contenedor on-the-fly y metemos la imagen
                open_grid = f'\n    <div class="plot-grid" style="{grid_style}">\n      {img_tag}<!-- PLOTS-INLINE-ANCHOR -->\n    </div>\n'
                html = html[:pos2] + open_grid + html[pos2:]

        html_path.write_text(html, encoding="utf-8")
        return str(img_path)
