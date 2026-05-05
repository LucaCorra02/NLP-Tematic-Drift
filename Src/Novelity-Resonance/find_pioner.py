from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PioneerAnalyzer:
    def __init__(self, similarity_path, data_path, metric_ris_path, graphics_path):
        self.similarity_path = Path(similarity_path)
        self.data_path = Path(data_path)
        self.metric_ris_path = Path(metric_ris_path)
        os.makedirs(graphics_path, exist_ok=True)
        self.graphics_path = Path(graphics_path)
        tmp_df_sim = pd.read_parquet(self.similarity_path)
        tmp_df_data =  pd.read_parquet(self.data_path)
        self.df_merged = pd.merge(
            tmp_df_sim,
            tmp_df_data[["id", "doi", "title", "abstract", "publication_year"]],
            on = "id",
            how = "inner"
        )
        self.df_merged = self.df_merged.sort_values(by=["publication_year"]).reset_index(drop=True)
        self.embeddings = np.vstack(self.df_merged["embedding"].values)
        self.similarity_matrix = cosine_similarity(self.embeddings)
        self.plot_df = None

    """
        I use KNN (K-Nearest-Neighbourhood instead of centroid for every years)
        Calculate some metrics as: 
        - Novelty = Distance between a specific paper and the papers published in n past years ( 1 - cosine_past)
        - Transience = Distance between a specific paper and the papers published in n future years (1 - cosine_future)
        - Resonance [-1,1] = Novelty - Transience 
    """
    def calculate_metrics(self, k_mean = 10 ,year_window = 2,):
        years = self.df_merged["publication_year"].unique()
        valid_years_mask = [True if year - year_window in years and year + year_window in years else False for year in years ]
        valid_years = years[valid_years_mask]
        all_paper_years = self.df_merged["publication_year"].values

        year_indices = {}
        for y in years:
            past_idx = np.where((all_paper_years >= y - year_window) & (all_paper_years < y))[0]
            future_idx = np.where((all_paper_years > y) & (all_paper_years <= y + year_window))[0]
            year_indices[y] = (past_idx, future_idx)

        results = {}
        for id_paper_row in range(len(self.df_merged)):
            year_paper = self.df_merged["publication_year"].iloc[id_paper_row]
            ids = self.df_merged.iloc[id_paper_row]["id"]
            if year_paper not in valid_years:
                results[ids] = {"Novelty": np.nan, "Transience": np.nan, "Resonance": np.nan}
                continue

            past_mask_index, future_mask_index = year_indices[year_paper]
            novelty = self._calculate_similarity_masked(id_paper_row, past_mask_index, k_mean)
            transience = self._calculate_similarity_masked(id_paper_row, future_mask_index, k_mean)
            resonance = novelty - transience # Sim_Future - Sim_Past
            results[ids] = {"Novelty": float(novelty), "Transience": float(transience), "Resonance": float(resonance)}

        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.reset_index(inplace=True)
        df_results.rename(columns={"index": "id"}, inplace=True)
        print("before: ", len(df_results))
        df_results = df_results.dropna()
        print("after: ",len(df_results))

        df_results['Novelty_Z'] = (df_results['Novelty'] - df_results['Novelty'].mean()) / df_results['Novelty'].std()
        df_results['Transience_Z'] = (df_results['Transience'] - df_results['Transience'].mean()) / df_results['Transience'].std()
        df_results['Resonance_Z'] = df_results['Novelty_Z'] - df_results['Transience_Z']
        df_results.to_csv(self.metric_ris_path, index=False)
        return df_results

    def _calculate_similarity_masked(self, id_paper: int, mask_index: np.array, k_mean):
        if len(mask_index) > 0:
            similarities = self.similarity_matrix[id_paper, mask_index]
            k = min(k_mean, len(similarities))
            top_k_similarity = np.mean(np.partition(similarities, -k)[-k:])
            return 1.0 - top_k_similarity
        return np.nan

    def plot_results(self):
        df_result = pd.read_csv(self.metric_ris_path)
        df_plot = pd.merge(df_result, self.df_merged, on="id", how="inner")
        self.plot_df = df_plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.scatterplot(data=df_plot, x='Novelty_Z', y='Transience_Z',
                        hue='publication_year', palette='viridis', alpha=0.6, ax=axes[0])
        axes[0].plot([df_plot['Novelty_Z'].min(), df_plot['Novelty_Z'].max()],
                     [df_plot['Novelty_Z'].min(), df_plot['Novelty_Z'].max()],
                     'r--', label='X=Y (Trade-off)')
        axes[0].set_title('Novelty vs Transience')
        axes[0].legend()

        yearly_res = df_plot.groupby('publication_year')['Resonance_Z'].mean().reset_index()
        sns.lineplot(data=yearly_res, x='publication_year', y='Resonance_Z', marker='o',
                     color='b', linewidth=2, ax=axes[1])
        axes[1].set_title('Average Resonance over Time')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Mean Resonance')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.graphics_path / "Scatter.jpeg")

    def plot_distribution(self):
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(self.plot_df["Novelty_Z"], bins=100)
        axs[1].hist(self.plot_df["Transience_Z"], bins=100)
        plt.show()

"""
    TODO: Graficare distribuzioni
    TODO: Prendere i top 10 paper e vedere di cosa parlano
    TODO: incrociare i dati con etichette bert topic
"""
if __name__ == "__main__":
    analysis = PioneerAnalyzer(
        similarity_path="../Embeddings/Similarity/similarity.parquet",
        data_path="../Data/Raw/scraped_data_cleaned.parquet",
        metric_ris_path = "metric.csv",
        graphics_path="Plots"
    )
    ris_df = analysis.calculate_metrics(k_mean=10, year_window=3)
    print("metrics_df: ", ris_df)
    analysis.plot_results()
    analysis.plot_distribution()