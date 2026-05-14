import os
import pandas as pd
from Src.Embeddings.similarity import PlotSimilarity
import numpy as np
import matplotlib.pyplot as plt
import ast
from collections import Counter
from bertopic import BERTopic
from scipy.stats import pearsonr, ttest_ind

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

        df_compare = pd.DataFrame(rows).sort_values("overrepresentation", ascending=False)
        print("over-representation concept:" ,df_compare[df_compare["overrepresentation"] > 0].head(20).to_string(index=False))
        return df_compare

    def cross_validate_with_topics(self, topic_model_path):
        model = BERTopic.load(topic_model_path)
        df_original = self.df_abstract.copy()
        assert len(model.topics_) == len(df_original)
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
        print(f"Papers in Topic -1: {n_minus1} ({n_minus1 / len(df) * 100:.1f}%)")
        print(f"Mean alignment (Topic -1): {mean_minus1:.4f}")
        print(f"Mean alignment (others): {mean_others:.4f}")
        print(f"Delta: {mean_others - mean_minus1:.4f}")
        print(f"Welch t-test: t={t_stat:.3f}, p={p_val:.6f}")

        outlier_ids = set(self.find_outliers()["id"])
        topic_minus1_ids = set(df[df["is_topic_minus1"]]["id"])
        overlap = outlier_ids & topic_minus1_ids

        print("outlier (alignment) intersect topic -1:")
        print(f"Outlier tot: {len(outlier_ids)}")
        print(f"Topic -1 tot: {len(topic_minus1_ids)}")
        print(f"Overlap: {len(overlap)} "
              f"({len(overlap) / max(len(outlier_ids), 1) * 100:.1f}% of the outlier)")
        return {
            "n_topic_minus1": n_minus1,
            "mean_topic_minus1": mean_minus1,
            "mean_others": mean_others,
            "p_value": p_val,
            "overlap_count": len(overlap),
            "overlap_pct": len(overlap) / max(len(outlier_ids), 1) * 100
        }

    """
        Needs to execute find_pioner.py first, need top_pioneer.csv
    """
    def pioneer_anlyze_score(self, pioneer_score_csv="top_pioneer.csv"):
        df_top_pioneer = pd.read_csv(pioneer_score_csv)
        df_merged = pd.merge(
            df_top_pioneer,
            self.df_score[["id", "alignment_score"]],
            on="id", how="inner"
        )
        df_merged.sort_values("alignment_score")
        df_merged.to_csv(os.path.join(self.output_dir, "pioneers_score.csv"), index=False)

        # alignment and resonance are linear correlated?
        r, p = pearsonr(df_merged["resonance"], df_merged["alignment_score"])
        print("\n Pioneers Analysis")
        print(f"Pearson r(Resonance, Alignment) = {r:.3f}, p = {p:.4f}, p <= 0.05? {p <= 0.05}")
        corpus_mean = self.df_score["alignment_score"].mean()
        pioneer_mean = df_merged["alignment_score"].mean()
        df_corpus_non_pioneer = self.df_score[~self.df_score["id"].isin(df_merged["id"])]

        # the differences between pioneer's alignment and other papers, counts?
        t_stat, p_val = ttest_ind(
            df_merged["alignment_score"],
            df_corpus_non_pioneer["alignment_score"],
            equal_var=False
        )
        print(f"Mean alignment - Pioneers: {pioneer_mean:.4f} | Corpus: {corpus_mean:.4f}, Delta: {corpus_mean - pioneer_mean:.4f}")
        print(f"Welch t-test: t={t_stat:.3f}, p={p_val:.4f}, p <= 0.05? {p_val <= 0.05}")

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            df_merged["resonance"], df_merged["alignment_score"],
            c=df_merged["year"], cmap="viridis", s=60, alpha=0.8
        )
        plt.colorbar(scatter, ax=ax, label="Publication Year")
        ax.set_xlabel("Resonance Z-score")
        ax.set_ylabel("Alignment Score")
        ax.set_title(f"Pioneer Papers: Resonance vs Scope Alignment\n(r={r:.3f}, p={p:.3f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pioneer_resonance_vs_alignment.png"), dpi=300)

        return {"pearson_r": r, "p_value_corr": p,
                "pioneer_mean_align": pioneer_mean, "corpus_mean_align": corpus_mean,
                "ttest_p": p_val}


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
    print(analyzer.pioneer_anlyze_score("../Novelity_Resonance/top_pioneer.csv"))

