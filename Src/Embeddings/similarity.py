import math

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics.pairwise import cosine_similarity
import umap

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

        df["max_scores"] = df.filter(like="score_").max(axis=1)
        years = sorted(df["publication_year"].astype(int).unique())
        data_to_plot = [
            df[df["publication_year"] == y]["max_scores"]
            for y in years
        ]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data_to_plot, patch_artist=True, positions=years)
        ax.set(xlabel='Year', ylabel='Max Scores Distribution',
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

    def plot_heat_matrix_embedding(self):
        df = self.df_merged
        years = sorted(df["publication_year"].unique().tolist())

        centroid = df.groupby("publication_year")["embedding"].mean()
        emb_matrix = np.vstack(centroid.values)
        score_matrix = cosine_similarity(emb_matrix)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(score_matrix)
        ax.set_xticks(range(len(years)), labels=years,
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(years)), labels=years)

        """
        for i in range(len(years)):
            for j in range(len(years)):
                text = ax.text(j, i, round(score_matrix[i, j],2),
                               ha="center", va="center", color="w")
        """

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Similarity Score", rotation=270, labelpad=15)
        ax.set_title("Embedding Centroid per Year")
        fig.tight_layout()
        plt.savefig(self.output_dir + "/heat_map.png")

    def heat_map_v2(self):
        df = self.df_merged
        years = sorted(df["publication_year"].unique().tolist())
        global_centroid = df["embedding"].mean()
        centroid = df.groupby("publication_year")["embedding"].mean()

        diff_centroid = [cent - global_centroid for cent in centroid.values]
        emb_matrix = np.vstack(diff_centroid)
        score_matrix = cosine_similarity(emb_matrix)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(score_matrix)
        ax.set_xticks(range(len(years)), labels=years,
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(years)), labels=years)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Similarity Score", rotation=270, labelpad=15)
        ax.set_title("Embedding Centroid per Year")
        fig.tight_layout()
        plt.savefig(self.output_dir + "/heat_map_v2.png")

    def plot_umap_2d(self):
        years, centroids_matrix = self._compute_centroids(self.df_merged)
        mapper = umap.UMAP(
            n_components=2,
            random_state=42,
            metric='cosine'
        ).fit(centroids_matrix)

        x_coords = mapper.embedding_[:, 0]
        y_coords = mapper.embedding_[:, 1]
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_coords, y_coords, c=years, cmap='Spectral', s=100)

        for i, year in enumerate(years):
            plt.annotate(str(year), (x_coords[i], y_coords[i]),
                         xytext=(5, 5), textcoords='offset points')

        plt.plot(x_coords, y_coords, linestyle=':', alpha=0.5, color='gray')
        plt.colorbar(scatter, label='Anno')
        plt.title("Centroids UMAP")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.output_dir + "/UMAP.png")

    """
        Calculate centroids for each years.
        Returns (years, centroids_matrix) with shape (N_years, 768).
    """
    def _compute_centroids(self, df):
        years = sorted(df["publication_year"].unique().tolist())
        centroids = []
        for year in years:
            vecs = np.vstack(df[df["publication_year"] == year]["embedding"].values)
            centroid = vecs.mean(axis=0)
            centroids.append(centroid)
        return years, np.vstack(centroids)

    def compute_k_mean_cosine(self, k_global_similar=500):
        df = self.df_merged
        years = sorted(df["publication_year"].unique().tolist())

        year_embeddings = {}
        for year in years:
            year_embeddings[year] = np.vstack(df[df["publication_year"] == year]["embedding"].values)
        ris_heatmap = {}
        for year_out in years:
            ris_heatmap[year_out] = {}
            year_out_emb = year_embeddings[year_out]
            for year_in in years:
                if year_out == year_in:
                    ris_heatmap[year_out][year_in] = np.nan
                    continue

                year_in_emb = year_embeddings[year_in]
                distance_matrix = cosine_similarity(year_out_emb, year_in_emb)
                flat_distances = distance_matrix.flatten()
                k_actual = min(k_global_similar, len(flat_distances))
                top_k_scores = np.sort(flat_distances)[-k_actual:]
                ris_heatmap[year_out][year_in] = np.mean(top_k_scores)

        df_heatmap = pd.DataFrame(ris_heatmap)
        score_matrix = df_heatmap.to_numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin = np.nanmin(score_matrix)
        vmax = np.nanmax(score_matrix)
        im = ax.imshow(score_matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(years)), labels=years, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(years)), labels=years)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Top {k_global_similar} Pairs Similarity", rotation=270, labelpad=15)
        ax.set_title("Symmetric Semantic Similarity Between Years")
        fig.tight_layout()
        plt.savefig(self.output_dir + "/hat-,ap-k-score.png")
        return score_matrix


    def heat_map_mmd(self, esp):
        df = self.df_merged
        years = sorted(df["publication_year"].unique().tolist())
        n_year = len(years)

        year_embeddings = {}
        for year in years:
            year_matrix = np.vstack(df[df["publication_year"] == year]["embedding"].values) # Embeddings matrix for every year
            year_embeddings[year] = year_matrix

        # Intra similarity for every year
        intra_similarity = {}
        for year, embedding_matrix in year_embeddings.items():
            matrix = cosine_similarity(embedding_matrix, embedding_matrix)
            matrix_no_diag = matrix.copy()
            np.fill_diagonal(matrix_no_diag, 0)
            intra_similarity[year] = np.sum(matrix_no_diag) / (len(matrix) * (len(matrix) - 1))

        mmd_matrix = np.zeros((n_year, n_year))
        for i, year_i in enumerate(years):
            for j, year_j in enumerate(years):
                if i == j:
                    mmd_matrix[i, j] = 0.0
                    continue

                matrix_year_i = year_embeddings[year_i]
                matrix_year_j = year_embeddings[year_j]
                inter_similarity = np.mean(cosine_similarity(matrix_year_i, matrix_year_j))
                # MMD^2 = SIM[Year_X] + SIM[Year_Y] - 2* SIM[Year_X, Year_Y]
                mmd_matrix[i,j] = intra_similarity[year_i] + intra_similarity[year_j] - 2 * inter_similarity

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(mmd_matrix, cmap="YlOrRd", aspect='auto')
        ax.set_xticks(range(n_year), labels=years, rotation=45, ha="right")
        ax.set_yticks(range(n_year), labels=years)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("MMD Distance", rotation=270, labelpad=20)
        ax.set_title("Distribution Shift over Time (MMD Squared)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Year")
        fig.tight_layout()
        plt.savefig(self.output_dir + "/heat_map_v2.png", dpi=300)
        print("✅ Salvato: heat_map_mmd_v2.png")
        plt.close()



if __name__ == "__main__":
    similarity = Similtarity("Emb/normalize_embedding.parquet", "Emb/scope_embeddings.parquet", "Similarity")
    similarity.calculate_cosine_similarity("similarity.parquet")

    plotsim = PlotSimilarity(
        "Similarity/similarity.parquet", "../Data/Raw/scraped_data_cleaned.parquet",
        "Emb/scope_embeddings.parquet", "Similarity"
    )
    """
    plotsim.plot_similarity_distribution()
    plotsim.plot_year_similarity()
    plotsim.plot_year_similarity_box()
    plotsim.plot_single_score_trend()
    plotsim.plot_heat_matrix_embedding()
    plotsim.heat_map_v2()
    plotsim.plot_umap_2d()
    """
    #plotsim.compute_k_mean_cosine(200)
    plotsim.heat_map_mmd(1e-20)