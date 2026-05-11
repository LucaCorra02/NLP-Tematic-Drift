import os
import pandas as pd
from similarity import PlotSimilarity
import numpy as np
import matplotlib.pyplot as plt
import ast
from collections import Counter
from bertopic import BERTopic
from scipy.stats import ttest_ind

 # TODO: add pioneer score check
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
        return outliers


    def analyze_outlier_concepts(self, outliers_csv_path="outliers_bottom.csv"):
        df_full = self.df_abstract.copy()
        df = pd.merge(
            self.df_score[["id", "alignment_score"]],
            df_full[["id", "concepts"]],
            on="id", how="inner"
        )
        df_outliers = pd.read_csv(os.path.join(self.output_dir, outliers_csv_path))
        bottom_ids = set(df_outliers["id"])

        def extract_l1_concept(concepts):
            if isinstance(concepts, np.ndarray):
                concepts = concepts.tolist()
            elif isinstance(concepts, str):
                concepts = ast.literal_eval(concepts)
            if not isinstance(concepts, list) or len(concepts) == 0:
                return None
            l1 = [c for c in concepts if isinstance(c, dict) and c.get("level") == 1]
            if not l1: return None
            return max(l1, key=lambda x: x.get("score", 0))["display_name"]

        df["top_concept"] = df["concepts"].apply(extract_l1_concept)
        n_parsed = df["top_concept"].notna().sum()
        print(f"Paper with L1 concept: {n_parsed}/{len(df)}")
        if n_parsed == 0: return None
        outlier_df = df[df["id"].isin(bottom_ids)]
        corpus_df = df[~df["id"].isin(bottom_ids)]

        outlier_counts = Counter(outlier_df["top_concept"].dropna()) # Count label freq
        corpus_counts = Counter(corpus_df["top_concept"].dropna())
        total_out = sum(outlier_counts.values())
        total_corp = sum(corpus_counts.values())
        all_concepts = set(outlier_counts) | set(corpus_counts) # get corpus in the intersection
        rows = []
        for c in all_concepts:
            pct_out = outlier_counts.get(c, 0) / total_out * 100
            pct_corp = corpus_counts.get(c, 0) / total_corp * 100
            rows.append({"concept": c, "pct_outlier": pct_out,
                         "pct_corpus": pct_corp, "overrepresentation": pct_out - pct_corp})

        df_compare = pd.DataFrame(rows).sort_values("pct_outlier", ascending=False)
        print("over-representation concept:" ,df_compare[df_compare["overrepresentation"] > 0].head(20).to_string(index=False))
        return df_compare

    def cross_validate_with_topics(self, topic_model_path):
        model = BERTopic.load(topic_model_path)
        df_original = self.df_abstract.copy()
        abstracts = df_original["abstract"].tolist()
        doc_info = model.get_document_info(abstracts)
        df_original = df_original.reset_index(drop=True)
        df_original["Topic_ID"] = doc_info["Topic"].values
        df = pd.merge(
            self.df_score[["id", "alignment_score"]],
            df_original[["id", "Topic_ID"]],
            on="id", how="inner"
        )
        df["is_topic_minus1"] = df["Topic_ID"] == -1
        n_minus1 = df["is_topic_minus1"].sum()
        mean_others = df[~df["is_topic_minus1"]]["alignment_score"].mean()
        mean_minus1 = df[df["is_topic_minus1"]]["alignment_score"].mean()
        t_stat, p_val = ttest_ind(
            df[~df["is_topic_minus1"]]["alignment_score"],
            df[df["is_topic_minus1"]]["alignment_score"],
            equal_var=False
        )

        print("Alignment score topic -1")
        print(f"Papers in Topic -1  : {n_minus1} ({n_minus1 / len(df) * 100:.1f}%)")
        print(f"Mean alignment (Topic -1) : {mean_minus1:.4f}")
        print(f"Mean alignment (others)   : {mean_others:.4f}")
        print(f"Differenza               : {mean_others - mean_minus1:.4f}")
        print(f"Welch t-test: t={t_stat:.3f}, p={p_val:.6f}")

        outlier_ids = set(self.find_outliers()["id"])
        topic_minus1_ids = set(df[df["is_topic_minus1"]]["id"])
        overlap = outlier_ids & topic_minus1_ids

        print("outlier (alignment) intersect topic -1:")
        print(f"  Outlier tot          : {len(outlier_ids)}")
        print(f"  Topic -1 tot         : {len(topic_minus1_ids)}")
        print(f"  Overlap                 : {len(overlap)} "
              f"({len(overlap) / max(len(outlier_ids), 1) * 100:.1f}% of the outlier)")
        return {
            "n_topic_minus1": n_minus1,
            "mean_topic_minus1": mean_minus1,
            "mean_others": mean_others,
            "p_value": p_val,
            "overlap_count": len(overlap),
            "overlap_pct": len(overlap) / max(len(outlier_ids), 1) * 100
        }


if __name__ == "__main__":
    analyzer = AnalyzeOutliers(
        "Similarity/similarity.parquet",
        "../Data/Raw/scraped_data_cleaned.parquet",
        "Emb/scope_embeddings.parquet",
        "Outliers"
    )
    analyzer.find_outliers()
    analyzer.analyze_outlier_concepts()
    analyzer.cross_validate_with_topics("BertTopic/topic_bert_parquet")

