from pathlib import Path
from unittest import result
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class PioneerAnalyzer:
    def __init__(self, similarity_path, data_path, metric_ris_path):
        self.similarity_path = Path(similarity_path)
        self.data_path = Path(data_path)
        self.metric_ris_path = Path(metric_ris_path)
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
        for id_paper in range(len(self.df_merged)):
            year_paper = self.df_merged["publication_year"][id_paper]
            if year_paper not in valid_years:
                results[id_paper] = {"Novelty": np.nan, "Transience": np.nan, "Resonance": np.nan}
                continue

            past_mask_index, future_mask_index = year_indices[year_paper]
            novelty = self._calculate_similarity_masked(id_paper, past_mask_index, k_mean)
            transience = self._calculate_similarity_masked(id_paper, future_mask_index, k_mean)
            resonance = novelty - transience # Sim_Future - Sim_Past
            results[id_paper] = {"Novelty": float(novelty), "Transience": float(transience), "Resonance": float(resonance)}

        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.reset_index(inplace=True)
        df_results.rename(columns={'index': 'id'}, inplace=True)
        print("before: ", len(df_results))
        df_results = df_results.dropna()
        print("after: ",len(df_results))
        df_results.to_csv(self.metric_ris_path, index=False)

    def _calculate_similarity_masked(self, id_paper: int, mask_index: np.array, k_mean):
        if len(mask_index) > 0:
            similarities = self.similarity_matrix[id_paper, mask_index]
            k = min(k_mean, len(similarities))
            top_k_similarity = np.mean(np.partition(similarities, -k)[-k:])
            return 1.0 - top_k_similarity
        return np.nan

if __name__ == "__main__":
    analysis = PioneerAnalyzer(
        similarity_path="../Embeddings/Similarity/similarity.parquet",
        data_path="../Data/Raw/scraped_data_cleaned.parquet",
        metric_ris_path = "metric.csv"
    )
    analysis.calculate_metrics(k_mean=30, year_window=3)