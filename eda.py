import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
import numpy as np

sns.set_style("whitegrid")

plt.rcParams.update({
    # Title and labels
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 18,

    # Axis border thickness
    "axes.linewidth": 3.0,

    # Tick labels and thickness
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 3.0,
    "ytick.major.width": 3.0,

    # KDE curve thickness
    "lines.linewidth": 3.2,

    # Render sharp on beamer
    "figure.dpi": 150,
})


class EDA:

    # Constructor
    def __init__(self):
        pass
    
    # Descriptive analysis: DA
    # -------------------------------------------------------------------
    def get_descriptive_analysis(self, failures, normal, features_numeric):
         # Compute comparison stats
        discriptive_analysis_table = []
        for col in features_numeric:
            if pd.api.types.is_numeric_dtype(features_numeric[col]) and col != 'UDI':
               row = {
                    "Feature": col,
                    "Mean (normal)":  round(normal[col].mean(), 1),
                    "Mean (failure)": round(failures[col].mean(), 1),

                    "Median (normal)":  round(normal[col].median(), 1),
                    "Median (failure)": round(failures[col].median(), 1),


                    "std (normal)":  round(normal[col].std(), 1),
                    "std (failure)": round(failures[col].std(), 1),

                    "p90-p10 (normal)"   : round(normal[col].quantile(0.90) - normal[col].quantile(0.10), 1),
                    "p90-p10 (faillure)" : round(failures[col].quantile(0.90) - failures[col].quantile(0.10), 1)
               }
               discriptive_analysis_table.append(row)

        # << Descriptive statistics >>
        comparison_da = pd.DataFrame(discriptive_analysis_table)
        # Save to CSV
        comparison_da.to_csv("feature_comparison.csv", index=False, sep=";")

    # Get plots
    # --------------------------------------------------------------------
    def get_plots(self, failures, normal, features):
        for col in features:
            self._plot_kde(failures, normal, col)
            self._plot_violin(failures, normal, col)
            self._plot_gaussian(failures, normal, col)

    def _kde_l1_distance(self, x_fail, x_norm, bandwidth=0.5):
        # reshape
        Xf = x_fail.reshape(-1, 1)
        Xn = x_norm.reshape(-1, 1)

        # fit
        kde_f = KernelDensity(bandwidth=bandwidth).fit(Xf)
        kde_n = KernelDensity(bandwidth=bandwidth).fit(Xn)

        # evaluate on shared grid
        grid = np.linspace(
            min(Xf.min(), Xn.min()),
            max(Xf.max(), Xn.max()),
            500
        ).reshape(-1, 1)

        pdf_f = np.exp(kde_f.score_samples(grid))
        pdf_n = np.exp(kde_n.score_samples(grid))

        # L1 distance
        D = np.trapz(np.abs(pdf_f - pdf_n), grid.flatten())

        return float(D)


    # Plot KDE
    # --------------------------------------------------------------------
    def _plot_kde(self, failures, normal, feature_name):
        filename = 'plots/kde/' + ''.join(c.lower() for c in feature_name if not c.isspace())

        # compute D metric (L1 distance)
        x_fail = failures[feature_name].astype(float).values
        x_norm = normal[feature_name].astype(float).values
        D_value = self._kde_l1_distance(x_fail, x_norm, bandwidth=0.5)

        plt.figure(figsize=(8, 4))

        # KDE curves
        sns.kdeplot(normal[feature_name], label="Normal", color="tab:blue", bw_adjust=1)
        sns.kdeplot(failures[feature_name], label="Failure", color="tab:orange", bw_adjust=1)

        plt.title(f"KDE: {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel("Density")

        # legend text below
        y_offset = -0.18
        plt.figtext(0.22, y_offset, "Normal", ha="left", fontsize=16,
                    fontweight="bold", color="tab:blue")
        plt.figtext(0.62, y_offset, "Failure", ha="left", fontsize=16,
                    fontweight="bold", color="tab:orange")

        # D metric placed even lower
        plt.figtext(0.5, -0.28,
                    f"L1 distance (D) = {D_value:.3f}",
                    ha="center", fontsize=14, fontweight="bold")

        # Save
        plt.savefig(filename + '_kde_plot.png', dpi=300, bbox_inches="tight")
        plt.close()


    # Violin
    # -------------------------------------------------------------------
        # Violin
    # -------------------------------------------------------------------
    def _plot_violin(self, failures, normal, feature_name):

        filename = 'plots/violin/' + ''.join(
            c.lower() for c in feature_name if not c.isspace()
        )

        # 1) Compute D metric (same inputs as KDE)
        x_fail = failures[feature_name].astype(float).values
        x_norm = normal[feature_name].astype(float).values
        D_value = self._kde_l1_distance(x_fail, x_norm, bandwidth=0.5)

        tmp = pd.DataFrame({
            feature_name: pd.concat([normal[feature_name], failures[feature_name]]).astype(float),
            "Condition": (
                ["Normal"] * len(normal) +
                ["Failure"] * len(failures)
            ),
        })

        plt.figure(figsize=(8, 4))

        sns.violinplot(
            data=tmp,
            x="Condition",
            hue="Condition",           # avoids future warnings
            y=feature_name,
            inner="box",   # adds a small boxplot inside the violin
            cut=0,         # avoids extending beyond min/max data
            palette={"Normal": "tab:blue", "Failure": "tab:orange"},
        )
        plt.legend([], [], frameon=False)

        plt.title(f"Violin: {feature_name}")
        plt.xlabel("Condition")
        plt.ylabel(feature_name)

        y_offset = -0.18
        plt.figtext(0.22, y_offset, "Normal",
                    ha="left", fontsize=16, fontweight="bold", color="tab:blue")
        plt.figtext(0.62, y_offset, "Failure",
                    ha="left", fontsize=16, fontweight="bold", color="tab:orange")

        plt.figtext(
            0.5,
            -0.28,
            f"L1 distance (D) = {D_value:.3f}",
            ha="center",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig(filename + "_violin_plot.png", dpi=300, bbox_inches="tight")
        plt.close()



    # Gaussian (normal PDF) plots
    # -------------------------------------------------------------------
    def _plot_gaussian(self, failures, normal, feature_name):
        # Requires: import numpy as np
        filename = 'plots/gaussian/' + ''.join(
            c.lower() for c in feature_name if not c.isspace()
        )

        import numpy as np

        # Range for x-axis
        x_min = min(normal[feature_name].min(), failures[feature_name].min())
        x_max = max(normal[feature_name].max(), failures[feature_name].max())
        if x_max == x_min:  # avoid zero width
            x_max = x_min + 1.0
        pad = 0.1 * (x_max - x_min)
        xs = np.linspace(x_min - pad, x_max + pad, 500)

        # Fit Gaussian: mean and std from the data
        mu_n, sigma_n = normal[feature_name].mean(), normal[feature_name].std()
        mu_f, sigma_f = failures[feature_name].mean(), failures[feature_name].std()

        def pdf(x, mu, sigma):
            return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                -0.5 * ((x - mu) / sigma) ** 2
            )

        plt.figure(figsize=(8, 4))

        plt.plot(xs, pdf(xs, mu_n, sigma_n), label="Normal", color="tab:blue")
        plt.plot(xs, pdf(xs, mu_f, sigma_f), label="Failure", color="tab:orange")

        plt.title(f"Gaussian fit: {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel("Density")

        # Color-coded legend labels
        y_offset = -0.18
        plt.figtext(
            0.22,
            y_offset,
            "Normal",
            ha="left",
            fontsize=16,
            fontweight="bold",
            color="tab:blue",
        )
        plt.figtext(
            0.62,
            y_offset,
            "Failure",
            ha="left",
            fontsize=16,
            fontweight="bold",
            color="tab:orange",
        )

        plt.tight_layout()
        plt.savefig(filename + "_gaussian_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
