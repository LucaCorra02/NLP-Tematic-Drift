import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sympy.physics.units import years


class Similtarity:
    def __init__(self, abstract_embedding_path, scope_embedding_path, output_dir):
        assert abstract_embedding_path.endswith(".parquet"), "correct input format"
        assert scope_embedding_path.endswith(".parquet"),  "correct output format"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_abstract = pd.read_parquet(abstract_embedding_path, engine='pyarrow')
        self.df_scope = pd.read_parquet(scope_embedding_path, engine='pyarrow')


    @staticmethod
    def __check_normalized_embeddings(abs_matrix):
        return [np.isclose(np.linalg.norm(row), 1.0, atol=1e-5) for row in abs_matrix]

    """
        Embeddings needs to be normalized, Cosine Similarity -> Dot Product
    """
    def calculate_cosine_similarity(self, output_file_name):
        abs_matrix = np.vstack(self.df_abstract["embedding"].values)
        scope_matrix = np.vstack(self.df_scope["embedding"].values)
        assert Similtarity.__check_normalized_embeddings(abs_matrix).count(True) == len(self.df_abstract)
        assert Similtarity.__check_normalized_embeddings(scope_matrix).count(True) == len(self.df_scope)
        assert abs_matrix.shape[1] == scope_matrix.T.shape[0], "Wrong Shape"

        similarity_matrix = abs_matrix @ scope_matrix.T
        scope_ids = self.df_scope["id"].tolist()
        score_cols = {
            f"score_{sid}": similarity_matrix[:, j]
            for j, sid in enumerate(scope_ids)
        }
        df_result = pd.DataFrame({
            "id": self.df_abstract["id"].values,
            "embedding": self.df_abstract["embedding"].values,
            **score_cols
        })
        df_result.to_parquet(str(os.path.join(self.output_dir, output_file_name)))
        return

class PlotSimilarity:
    def __init__(self, score_input_path, abstract_input_path, scope_embedding_path, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_abstract = pd.read_parquet(abstract_input_path, engine='pyarrow')
        self.df_score = pd.read_parquet(score_input_path, engine='pyarrow')
        self.df_merged = pd.merge(self.df_score, self.df_abstract[['id', 'publication_year']], on='id', how='left')
        assert self.df_merged.isna().sum().sum() == 0, "nan values"
        self.df_scope = pd.read_parquet(scope_embedding_path, engine='pyarrow')

    def plot_similarity_distribution(self):
        df = self.df_score
        scores_col = df.loc[:,'score_scope-1':]
        df["mean_scores"] = scores_col.mean(axis=1)
        data = df["mean_scores"]
        plt.hist(data, bins=40, color='skyblue', edgecolor='black')
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.title("Score Distribution")
        plt.savefig(self.output_dir + "/distribution.png")

    def plot_year_similarity(self):
        df = self.df_merged
        df["mean_scores"] = df.filter(like="score_").mean(axis=1)
        mean_per_year = df.groupby("publication_year")["mean_scores"].mean()
        years = [year for year in mean_per_year.keys()]
        mean = [mean_per_year[year] for year in mean_per_year.keys()]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, mean, marker='o', linestyle='-')
        ax.set(xlabel='Year', ylabel='Mean Score',
               title='Similarity trend')
        ax.grid()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=45)
        fig.savefig(self.output_dir + "/trend.png")

    def plot_year_similarity_box(self):
        df = self.df_merged

        df["mean_scores"] = df.filter(like="score_").mean(axis=1)
        years = sorted(df["publication_year"].astype(int).unique())
        data_to_plot = [
            df[df["publication_year"] == y]["mean_scores"]
            for y in years
        ]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data_to_plot, patch_artist=True, positions=years)
        ax.set(xlabel='Year', ylabel='Mean Scores Distribution',
               title='Similarity Distribution per Year')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(self.output_dir + "/trend_boxplot.png")

    def plot_single_score_trend(self):
        df = self.df_merged
        scopes = df.filter(like="score_").columns.values.tolist()
        years = sorted(df["publication_year"].unique().tolist())
        df_by_year = df.groupby("publication_year")
        ris = {}
        for scope in scopes:
            df_mean = df_by_year[scope].mean()
            ris[scope] = df_mean.values.tolist()

        plt.figure(figsize=(10, 6))
        for key, value in ris.items():
            plt.plot(years, value, marker='o', label=key, linewidth=2, markersize=6)

        min_year = int(min(years))
        max_year = int(max(years))
        plt.xticks(range(min_year, max_year + 1))
        plt.tick_params(axis='x', rotation=45)
        plt.title('Score trend per year', fontsize=14, pad=15)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Scope', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir + "/trend_single_scope.png")


if __name__ == "__main__":
    similarity = Similtarity("Emb/normalize_embedding.parquet", "Emb/scope_embeddings.parquet", "Similarity")
    similarity.calculate_cosine_similarity("similarity.parquet")

    plotsim = PlotSimilarity(
        "Similarity/similarity.parquet", "../Data/Raw/scraped_data_cleaned.parquet",
        "Emb/scope_embeddings.parquet", "Similarity"
    )
    plotsim.plot_similarity_distribution()
    plotsim.plot_year_similarity()
    plotsim.plot_year_similarity_box()
    plotsim.plot_single_score_trend()
