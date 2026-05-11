import os
import pandas as pd
from similarity import PlotSimilarity

 # TODO: add BertTopic -1 check
class AnalyzeOutliers:
    """
        Require to execute similarity.py first
    """
    def __init__(self, score_input_path, abstract_input_path, scope_embedding_path, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_abstract = pd.read_parquet(abstract_input_path, engine='pyarrow') # df with paper info
        plotsim = PlotSimilarity(
            score_input_path,
            abstract_input_path,
            scope_embedding_path,
            "Similarity"
        )
        self.df_score = plotsim.compute_unified_score() # df with alignement score

    def find_outliers(self):
        df = pd.merge(
            self.df_score[["id", "alignment_score", "best_scope", "publication_year"]],
            self.df_abstract[["id", "title", "abstract"]],
            on="id", how="inner"
        )
        Q1 = df["alignment_score"].quantile(0.25)
        Q3 = df["alignment_score"].quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q1 - 1.5 * IQR
        outliers = df[df["alignment_score"] < threshold]
        outliers.to_csv(os.path.join(self.output_dir, "outliers_bottom.csv"), index=False)


if __name__ == "__main__":
    analyzer = AnalyzeOutliers(
        "Similarity/similarity.parquet",
        "../Data/Raw/scraped_data_cleaned.parquet",
        "Emb/scope_embeddings.parquet",
        "Outliers"
    )
    analyzer.find_outliers()

