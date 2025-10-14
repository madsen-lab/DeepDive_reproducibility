from sklearn.decomposition import TruncatedSVD
import numpy as np
from scib_metrics.nearest_neighbors import NeighborsResults
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from plottable import ColumnDefinition, Table


import os

def lsi(adata):
    X = adata.X
    tf = X.multiply(1 / X.sum(axis=1))  
    idf = np.log(1 + X.shape[0] / (1 + X.sum(axis=0))).A1  
    tfidf = tf.multiply(idf)
    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi = svd.fit_transform(tfidf)
    adata.obsm["X_lsi"] = lsi


#From https://github.com/YosefLab/scib-metrics/blob/42107f2450943aba526db3dd23434f49fef8a1fa/src/scib_metrics/benchmark/_core.py
_LABELS = "labels"
_BATCH = "batch"
_X_PRE = "X_pre"
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"

def plot_results_table(self, min_max_scale: bool = False, show: bool = True, save_dir: str | None = None, dataset:str = '') -> Table:
        """Plot the benchmarking results.

        Parameters
        ----------
        min_max_scale
            Whether to min max scale the results.
        show
            Whether to show the plot.
        save_dir
            The directory to save the plot to. If `None`, the plot is not saved.
        """
        num_embeds = len(self._embedding_obsm_keys)
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
        df = self.get_results(min_max_scale=min_max_scale)
        # Do not want to plot what kind of metric it is
        plot_df = df.drop(_METRIC_TYPE, axis=0)
        # Sort by total score
        if self._batch_correction_metrics is not None and self._bio_conservation_metrics is not None:
            sort_col = "Total"
        elif self._batch_correction_metrics is not None:
            sort_col = "Batch correction"
        else:
            sort_col = "Bio conservation"
        plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index

        # Split columns by metric type, using df as it doesn't have the new method col
        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
        column_definitions = [
            ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
        ]
        # Circles for the metric values
        column_definitions += [
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for i, col in enumerate(other_cols)
        ]
        # Bars for the aggregate scores
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]
        # Allow to manipulate text post-hoc (in illustrator)
        with mpl.rc_context({"svg.fonttype": "none", 'pdf.fonttype':42, 'ps.fonttype':42}):
            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
            tab = Table(
                plot_df,
                cell_kw={
                    "linewidth": 0,
                    "edgecolor": "k",
                },
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": 10, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        if show:
            plt.show()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f"{dataset}_scib_results.pdf"), facecolor=ax.get_facecolor(), dpi=300)

        return tab