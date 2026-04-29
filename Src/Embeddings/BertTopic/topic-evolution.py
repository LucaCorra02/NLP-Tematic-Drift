from unittest import result
import pandas as pd
from bertopic import BERTopic
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import html
import re
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import os
import pymannkendall as mk
from scipy.stats import theilslopes
from pathlib import Path

class BertTopic:
    def __init__(self, embeding_with_score_path, abstract_embedding_path, save_model_file_name):
        self.df_abstract = pd.read_parquet(abstract_embedding_path ,engine='pyarrow')
        self.df_score = pd.read_parquet(embeding_with_score_path, engine='pyarrow')
        self.df_merged = pd.merge(self.df_score, self.df_abstract[['id', 'abstract', 'publication_year']], on='id', how='left')
        self.df_merged["abstract"] = [self._parse_string(abs) for abs in self.df_merged["abstract"].tolist()]
        self.save_model_file_name = save_model_file_name
        assert self.df_merged.isna().sum().sum() == 0, "nan values"
        vectorizer_model = CountVectorizer(
            stop_words="english", min_df=2, ngram_range=(1, 2)
        )
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        self.model = BERTopic(
            language="english",
            verbose=True,
            top_n_words=10,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True
        )
        print(len(self.df_merged))

    @staticmethod
    def _parse_string(text: str):
        text = html.unescape(text)
        text = re.sub(r'\s*</?(?:sub|sup)[^>]*>\s+([^A-Z])', r' \1', text)
        text = re.sub(r'\s*</?(sub|sup)[^>]*>\s*', "", text, flags=re.IGNORECASE)
        text = re.sub(r'</?(i|b|em|strong)[^>]*>', "", text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', "", text)
        text = re.sub(r'https?:\/\/.\S+', "", text)
        text = re.sub(r'www\.\S+', "", text)
        prefixes_to_remove = r'^(Abstract\.|Abstract|Summary\.|Background:|Introduction:|Methods:|Results:|Conclusion:|Conclusions:)\s*'
        text = re.sub(prefixes_to_remove, "", text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', " ", text).strip()
        return text

    def get_dataframe_paper(self):
        return self.df_merged

    def train_topicbert(self):
        if os.path.exists(self.save_model_file_name):
            print("Model already trained")
            self.model = BERTopic.load(self.save_model_file_name)
            return

        df = self.df_merged
        abstract = [self._parse_string(abs) for abs in df["abstract"].tolist()]
        embeddings = np.vstack(df["embedding"].values)
        #years = df["publication_year"].tolist()
        topics, probs = self.model.fit_transform(abstract, embeddings=embeddings)

        print("Before Outlier step")
        topic_info_before = self.model.get_topic_info()
        print(topic_info_before.head(10))
        outliers_num = topic_info_before[topic_info_before["Topic"] == -1]["Count"].values[0]
        print(f"Papers with Topic -1: {outliers_num} / {len(abstract)}")
        self.reduce_outliers(abstract, topics, probs, embeddings)
        self.model.save(self.save_model_file_name, serialization="pickle", save_ctfidf=True)

        fig = self.model.visualize_documents(abstract, embeddings=embeddings, hide_document_hover=False,
                                             hide_annotations=True)
        fig.write_image("mappa_documenti.jpeg")
        fig = self.model.visualize_heatmap()
        fig.write_image("heatmap_documenti.jpeg")
        fig = self.model.visualize_hierarchy()
        fig.write_image("hierarchy_documenti.jpeg")
        print("Saved graphics")

    def reduce_outliers(self, abstract, topics, probs, embeddings):
        topics_step_1 = self.model.reduce_outliers(
            abstract,
            topics,
            probabilities=probs,
            strategy="probabilities",
            threshold=0.05
        )
        """
        new_topics = self.model.reduce_outliers(
            abstract,
            topics_step_1,
            strategy="embeddings",
            embeddings=embeddings
        )
        """
        self.model.update_topics(abstract, topics= topics_step_1 , vectorizer_model=self.model.vectorizer_model)
        print("After outliers removal")
        topic_info_after = self.model.get_topic_info()
        print(topic_info_after.head(10))
        outliers_num = 0
        if -1 in topic_info_after["Topic"].values:
            outliers_num = topic_info_after[topic_info_after["Topic"] == -1]["Count"].values[0]
        print(f"After outliers removal: {outliers_num} / {len(abstract)}")

        topics_words, diversity = self.calculate_diversity()
        coherence = self.calculate_coherence(topics_words, abstract)
        print(diversity, coherence)

        return diversity, coherence

    """
        It measures how musch a topic use unique words from the other. It's a external cluster metrics
    """
    def calculate_diversity(self):
        topic_info = self.model.get_topic_info()
        topics_words = []
        all_words = []
        for topic in topic_info['Topic']:
            if topic != -1:
                words = [word for word, _ in self.model.get_topic(topic)]
                topics_words.append(words)
                all_words.extend(words)

        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words) if all_words else 0
        print(f"Topic Diversity: {diversity:.4f}")
        return topics_words, diversity

    """
        It measures how much a topic has coherence words. It's a internal cluster metrics 
    """
    def calculate_coherence(self, topics_words, abstract_list):
        tokenized_docs = [doc.split() for doc in abstract_list]
        dictionary = Dictionary(tokenized_docs)
        cm = CoherenceModel(
            topics=topics_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        print(f"Topic Coherence: {coherence:.4f}")
        return coherence

class TopicEvolution:
    def __init__(self, save_model_file_name, df_paper: pd.DataFrame, topic_over_time_path):
        assert  os.path.exists(save_model_file_name)
        self.model = BERTopic.load(save_model_file_name)
        self.df = df_paper
        self.years = self.df["publication_year"].tolist()
        self.topics_over_time_path = Path(topic_over_time_path)
        if os.path.exists(self.topics_over_time_path):
            self.topics_over_time_df = pd.read_csv(self.topics_over_time_path / "topic_over_time.csv")
        else:
            topics_over_time = self.model.topics_over_time(self.df["abstract"], self.years)
            self.topics_over_time_df = topics_over_time
            self.topics_over_time_path.mkdir(parents=True, exist_ok=True)
            self.topics_over_time_df.to_csv(self.topics_over_time_path / "topic_over_time.csv")

    def topic_drift(self):
        df = self.topics_over_time_df
        papers_per_year = df.groupby("Timestamp")["Frequency"].sum()
        df["Freq_norm"] = df.apply(
            lambda row: row["Frequency"] / papers_per_year[row["Timestamp"]], axis=1
        )
        print(df)

        metrics = []
        for topic_id in sorted(df["Topic"].unique()):
            if topic_id == -1: continue
            topic_data = df[df["Topic"] == topic_id].sort_values("Timestamp").reset_index(drop=True)
            if len(topic_data) < 5:
                print(f"Topic {topic_id} removed")
                continue
            keywords_per_year = topic_data["Words"].tolist()
            result = self.analyze_single_topic(topic_id, topic_data, keywords_per_year)
            metrics.append(result)

        metrics_df = pd.DataFrame(metrics)
        thresholds = self.discover_thresholds(metrics_df)

        results = []
        for idx, row in metrics_df.iterrows():
            topic_id = int(row["topic_id"])
            topic_data = df[df["Topic"] == topic_id].sort_values("Timestamp")
            y_freq = topic_data["Freq_norm"].values.astype(float)

            classification = self.classify_topic_lifecycle(
                burstiness=row["burstiness"],
                trend_p_value=row["trend_p_val"],
                trend_slope=row["trend_slope"],
                volatility=row["volatility_val"],
                avg_jaccard=row["jaccard_mean"],
                drift_slope=row["lessical_slope"],
                thresholds=thresholds,
                frequencies = y_freq
            )
            result = {
                "topic_id": topic_id,
                "trend_classification": classification["trend_classification"],
                "lifecycle_score": classification["lifecycle_score"],
                "lexical_classification": classification["lexical_classification"],

                "burstiness": row["burstiness"],
                "burstiness_threshold": "High" if row['burstiness'] > thresholds['burstiness_p75'] else "Low",

                "trend_p_val": row["trend_p_val"],
                "trend_Slope": row["trend_slope"],
                'significant_trend': row["significant_trend"],

                "volatility_val": row["volatility_val"],
                'Volatility_vs_P75': "High" if row['volatility_val'] > thresholds['volatility_p75'] else "Low",
                "direction_changes": row["direction_changes"],

                "avg_Jaccard": row["jaccard_mean"],
                "Jaccard_vs_P75": "High" if not np.isnan(row["jaccard_mean"]) and row["jaccard_mean"] > thresholds[
                    'jaccard_p75'] else "Low",
                "lessical_slope": row["lessical_slope"],
                "jaccard_list": row["jaccard_list"],
                "n_years": row["n_years"],
                "freq_mean": row["freq_mean"],
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('lifecycle_score', ascending=False).reset_index(drop=True)
        results_df.to_csv(self.topics_over_time_path / "result_df_tmp.csv")
        return results_df

    def analyze_single_topic(self, topic_id, topic_data: pd.DataFrame, keywords_per_year: list):
        y_freq = topic_data["Freq_norm"].values.astype(float)
        burstiness = self.calculate_burstiness(y_freq)
        trend = self.calculate_trend(y_freq)
        volatility = self.calculate_volatility(y_freq)
        jaccard = self.calculate_jaccard(keywords_per_year)

        return {
            "topic_id": topic_id,
            "n_years": len(topic_data),
            "freq_mean": float(np.mean(y_freq)),
            "burstiness": burstiness["value"],
            "significant_trend": trend["is_significant"],
            "trend_p_val": trend["p_value"],
            "trend_slope": trend["slope"],
            "volatility_val": volatility["volatility_val"],
            "direction_changes": volatility["direction_changes"],
            "jaccard_list": jaccard["jaccard"],
            "jaccard_mean": jaccard["jaccard_avg"],
            "lessical_slope": jaccard["drift_slope"],
            "n_jaccard_pairs": jaccard["pairs"]
        }

    def discover_thresholds(self, metrics: pd.DataFrame) -> dict:
        burstiness_valid = metrics['burstiness'].dropna()
        volatility_valid = metrics['volatility_val'].dropna()
        jaccard_valid = metrics['jaccard_mean'].dropna()
        drift_slope_valid = metrics['lessical_slope'].dropna()

        thresholds = {
            'burstiness_p25': float(burstiness_valid.quantile(0.25)),
            'burstiness_p50': float(burstiness_valid.quantile(0.50)),
            'burstiness_p75': float(burstiness_valid.quantile(0.75)),

            'volatility_p25': float(volatility_valid.quantile(0.25)),
            'volatility_p50': float(volatility_valid.quantile(0.50)),
            'volatility_p75': float(volatility_valid.quantile(0.75)),

            'jaccard_p25': float(jaccard_valid.quantile(0.25)),
            'jaccard_p50': float(jaccard_valid.quantile(0.50)),
            'jaccard_p75': float(jaccard_valid.quantile(0.75)),

            'drift_p25': float(drift_slope_valid.quantile(0.25)),
            'drift_p50': float(drift_slope_valid.quantile(0.50)),
            'drift_p75': float(drift_slope_valid.quantile(0.75)),
        }
        print(thresholds)
        return thresholds

    def classify_topic_lifecycle( self, burstiness, trend_p_value, trend_slope, volatility, avg_jaccard, drift_slope, thresholds: dict, frequencies):
        if trend_p_value < 0.05:
            if trend_slope > 0:
                max_idx = np.argmax(frequencies)
                last_idx = len(frequencies) - 1
                if max_idx < last_idx - 2:  # peak not in last 3 years
                    classification = "Episodic (Past Peak)"
                    score = 0.3
                else:
                    classification = "Emergent"
                    score = 1.0
            else:
                classification = "Declining"
                score = 0.1
        else:
            if volatility > thresholds['volatility_p75']:
                if burstiness > thresholds['burstiness_p75']:
                    classification = "Episodic with spikes"
                    score = 0.75
                else:
                    classification = "Roller Coster"
                    score = 0.55
            else:
                classification = "Stable"
                score = 0.4

        if not np.isnan(avg_jaccard):
            if avg_jaccard > thresholds['jaccard_p75']:
                lexical_label = "stable keywords"
            elif avg_jaccard > thresholds['jaccard_p25']:
                lexical_label = "slowly changing keywords"
            else:
                lexical_label = "keywords are different"
        else:
            lexical_label = "no data"

        return {
            "trend_classification": classification,
            "lexical_classification": lexical_label,
            "lifecycle_score": score
        }

    """
        Measure if a trend is monotonic
        Formula: (sigma - mean) / (sigma + mean)
        Range: [-1,1]
    """
    def calculate_burstiness(self, frequencies):
        mean = np.mean(frequencies)
        std = np.std(frequencies)
        burstiness = 0.0
        if mean > 1e-10:
            #burstiness = (std - mean) / (std + mean)
            burstiness = std / mean

        return {'value': burstiness}

    """
        Mann-Kendall + Theil-Sen
    """
    def calculate_trend(self, frequencies: np.ndarray, alpha=0.05):
        mk_ris = mk.original_test(frequencies)
        p_val = float(mk_ris.p)
        slope, intercept, low, high = theilslopes(frequencies, np.arange(len(frequencies)), 0.95)
        return {
            'is_significant': p_val < alpha,
            'p_value': p_val,
            'slope': float(slope),
            'splope_low': float(low),
            'splope_high': float(high),
        }

    """
        Topic volatility, based on change of sings(direction)
        Formula: change of sign / (tot_year - 1)
        Range: [0,1]
    """
    def calculate_volatility(self, frequencies: np.ndarray):
        signs = np.sign(np.diff(frequencies))
        sign_changes = np.diff(signs)
        direction_changes = np.sum(sign_changes != 0)

        volatility = direction_changes / (len(frequencies) - 1)
        return {
            'volatility_val': float(volatility),
            'direction_changes': int(direction_changes),
            'total_period': len(frequencies)-1
        }

    """
        Formula: | A intersect B | / | A union B |
        Range: [0,1]
    """
    def calculate_jaccard(self, keywords_per_year: list):
        jaccard_list = []
        for i in range(1, len(keywords_per_year)):
            key_prev = set([w.strip() for w in keywords_per_year[i-1].split(",") if w.strip()])
            key_next =  set([w.strip() for w in keywords_per_year[i].split(",") if w.strip()])
            intersection = len(key_prev & key_next)
            union = len(key_prev | key_next)
            jaccard = intersection / union if union > 0 else np.nan
            jaccard_list.append(jaccard)

        if len(jaccard_list) >= 3:
            jaccard_array = np.array(jaccard_list)
            drift_slope, _, _, _ = theilslopes(
                jaccard_array,
                np.arange(len(jaccard_array)),
                0.95
            )
            avg_jaccard = np.mean(jaccard_list)
        else:
            drift_slope = np.nan
            avg_jaccard = np.nan

        return {
            "jaccard": [float(v) for v in jaccard_list],
            "jaccard_avg": avg_jaccard,
            "pairs": len(jaccard_list),
            "drift_slope": drift_slope,
        }

    def topic_evolution_graphic(self, top_n_topics = 20):
        fig = self.model.visualize_topics_over_time(
            self.topics_over_time_df,
            top_n_topics=top_n_topics
        )
        fig.show()
        fig.write_image("topics_over_time.jpeg")
        fig.write_html("topics_over_time.html")

if __name__ == "__main__":
    """
        If you changes hyper-parameters remove topic_bert_parquet model and Topic_Over_Time folder. Then refit the model
    """
    bertTop = BertTopic(
        "../Similarity/similarity.parquet",
        "../../Data/Raw/scraped_data_cleaned.parquet",
        "topic_bert_parquet"
    )
    bertTop.train_topicbert()
    topicEvolution = TopicEvolution("topic_bert_parquet", bertTop.get_dataframe_paper(), "Topic_Over_Time")
    topicEvolution.topic_drift()
    topicEvolution.topic_evolution_graphic()