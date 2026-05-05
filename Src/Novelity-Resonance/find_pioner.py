from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class PioneerAnalyzer:
    def __init__(self, similarity_path, data_path):
        self.similarity_path = Path(similarity_path)
        self.data_path = Path(data_path)
        tmp_df_sim = pd.read_parquet(self.similarity_path)
        tmp_df_data =  pd.read_parquet(self.data_path)
        self.df_merged = pd.merge(
            tmp_df_sim,
            tmp_df_data[["id", "doi", "title", "abstract", "publication_year"]],
            on = "id",
            how = "inner"
        )
        self.embeddings = np.vstack(self.df_merged["embedding"].values)
        self.similarity_matrix = cosine_similarity(self.embeddings)

if __name__ == "__main__":
    analysis = PioneerAnalyzer(
        similarity_path="../Embeddings/Similarity/similarity.parquet",
        data_path="../Data/Raw/scraped_data_cleaned.parquet"
    )