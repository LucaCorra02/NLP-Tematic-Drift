import os
import pandas as pd
from similarity import PlotSimilarity


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


if __name__ == "__main__":
    analyzer = AnalyzeOutliers(
        "Similarity/similarity.parquet",
        "../Data/Raw/scraped_data_cleaned.parquet",
        "Emb/scope_embeddings.parquet",
        "Outliers"
    )
    print(analyzer.df_score)

