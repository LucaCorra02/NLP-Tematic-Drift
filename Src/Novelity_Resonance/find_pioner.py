from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from bertopic import BERTopic

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
        axs[0].set_title("Novelty (Normalized)")
        axs[1].set_title("Transience (Normalized)")
        plt.savefig(self.graphics_path / "Distribution.jpeg")

    def print_top_pioneer(self, top_csv_path, threshold = 1.5):
        df = self.plot_df.copy()
        df.sort_values(by="Resonance_Z", ascending=False, inplace=True)
        df = df[df["Resonance_Z"] > threshold]
        num_pioneers = len(df)
        records = []
        for index, row in df.iterrows():
            record = {
                "id": row["id"],
                "year": row["publication_year"],
                "resonance": row["Resonance_Z"],
                "title": row["title"],
                "abstract": row["abstract"]
            }
            records.append(record)
            #print(f"Id:{row["id"]}, title:{row["title"]}, abstract:{row["abstract"]}, year:{row["publication_year"]}, resonance:{row["Resonance_Z"]}")
        ris = pd.DataFrame(records)
        ris.to_csv(top_csv_path)
        return ris, num_pioneers

    def analyze_pioneer_drivers(self, csv_path="top_pioneer.csv", top_n_papers=50):
        df = pd.read_csv(csv_path)
        top_pioneers = df.sort_values(by='resonance', ascending=False).head(top_n_papers)

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            return text

        abstracts = top_pioneers['abstract'].apply(clean_text).tolist()
        vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        words_df = pd.DataFrame({'word': feature_names, 'weight': avg_tfidf})
        words_df = words_df.sort_values(by='weight', ascending=False).head(20)
        print(words_df.to_string(index=False))


    def link_pioneer_and_topic_label(self, topic_bert__path, pioneer_csv_path = "top_pioneer.csv"):
        model = BERTopic.load(topic_bert__path)
        top_pioneer_df = pd.read_csv(pioneer_csv_path)
        originally_df = pd.read_parquet(self.data_path)
        assert len(model.topics_) == len(originally_df)
        originally_df["Topic_ID"] = model.topics_  # df_originally should be in the same order as the one use for bert topic train
        df_original = originally_df.copy()

        df_labeled = pd.merge(
            top_pioneer_df,
            originally_df[['id', 'Topic_ID']],
            on="id",
            how="inner"
        )
        topic_info = model.get_topic_info()
        df_labeled = pd.merge(
            df_labeled,
            topic_info[["Topic", "Name"]],
            left_on="Topic_ID",
            right_on="Topic",
            how="left"
        )
        return df_labeled, df_original

    """
        limits parameters help with the readability of the pie chart
    """
    def analyze_bert_topic_label(self, topic_bert__path, pioneer_csv_path ="top_pioneer.csv", limits = 20):
        df_labeled, _ = self.link_pioneer_and_topic_label(topic_bert__path, pioneer_csv_path)
        df_labeled = df_labeled[df_labeled["Topic_ID"] != -1]
        df_labeled[["id", "Topic", "Name", "title"]].to_csv("pioneer_with_label.csv")
        df_labeled = df_labeled.head(limits)
        topic_counts = df_labeled['Name'].value_counts()

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.Set3(range(len(topic_counts)))
        wedges, texts, autotexts = ax.pie(
            topic_counts.values,
            labels=topic_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        ax.set_title(f"Distribution of top {limits} pioneers by topic", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.graphics_path / "pioneers_pie_chart.jpeg", dpi=300, bbox_inches='tight')
        return topic_counts

    def dynamic_topic_evolution(self, topic_bert__path, pioneer_csv_path, topics_count):
        df_pioner, df_original = self.link_pioneer_and_topic_label(topic_bert__path, pioneer_csv_path)
        model = BERTopic.load(topic_bert__path)
        abstracts = df_original["abstract"].values
        years = df_original["publication_year"].values
        assert len(abstracts) == len(years)

        top_frequency = topics_count.head(5).keys()
        name_list = [name for name in top_frequency]
        print(name_list)
        ids = df_pioner[df_pioner["Name"].isin(name_list)]["Topic"]
        topics_over_time = model.topics_over_time(
            abstracts,
            years
        )
        fig = model.visualize_topics_over_time(
            topics_over_time,
            topics= ids,
            height=600
        )
        fig.write_html(self.graphics_path / "topics_over_time_dynamic.html")
        fig.write_image(self.graphics_path / "topics_over_time_dynamic.png")
"""
TODO: ADD TRAENCECY AND NOVELITY
TODO: ADD LOWEST SCORE RESONANCE
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
    ris, num_pioneers = analysis.print_top_pioneer("top_pioneer.csv", 1.5)
    print(ris)
    #analysis.analyze_pioneer_drivers("top_pioneer.csv", num_pioneers)
    topic_counts = analysis.analyze_bert_topic_label("../Embeddings/BertTopic/topic_bert_parquet", "top_pioneer.csv")
    analysis.dynamic_topic_evolution("../Embeddings/BertTopic/topic_bert_parquet", "top_pioneer.csv", topic_counts)