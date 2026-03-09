import pandas as pd
import os
import numpy as np

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



if __name__ == "__main__":
    similarity = Similtarity("Emb/normalize_embedding.parquet", "Emb/scope_embeddings.parquet", "Similarity")
    similarity.calculate_cosine_similarity("similarity.parquet")
    df = pd.read_parquet("Similarity/similarity.parquet")
