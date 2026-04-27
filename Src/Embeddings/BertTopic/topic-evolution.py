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

class TopicEvolution:
    def __init__(self, embeding_with_score_path, abstract_embedding_path, save_model_file_name):
        self.df_abstract = pd.read_parquet(abstract_embedding_path ,engine='pyarrow')
        self.df_score = pd.read_parquet(embeding_with_score_path, engine='pyarrow')
        self.df_merged = pd.merge(self.df_score, self.df_abstract[['id', 'abstract', 'publication_year']], on='id', how='left')
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
        self.model.save(self.save_model_file_name, serialization="safetensors", save_ctfidf=True)

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


if __name__ == "__main__":
    topicEvolution = TopicEvolution(
        "../Similarity/similarity.parquet",
        "../../Data/Raw/scraped_data_cleaned.parquet",
        "topic_bert_parquet"
    )
    topicEvolution.train_topicbert()
